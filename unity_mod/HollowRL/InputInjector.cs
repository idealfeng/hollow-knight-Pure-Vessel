using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using InControl;
using UnityEngine;

// Deliberately in the global namespace: the plugin class in connection.cs is
// itself called HollowRL, so a namespace of that name would make every
// reference ambiguous.

/// <summary>
/// Injects input into the game from inside the game process.
///
/// The trainer used to drive the knight with pyautogui - OS-level fake
/// keystrokes. That is global (every keystroke goes to whatever is focused,
/// which on a bad day is your browser), it needs the window focused, and it
/// is X11-only, which is a problem on a Wayland desktop.
///
/// Here we sit underneath all of that. The game reads its input through
/// InControl PlayerActions (HeroActions: left, jump, attack, ...); we
/// Harmony-patch the getters it reads them through and answer with the
/// trainer's intent for the actions it is currently driving. Nothing outside
/// this process ever sees an input event.
///
/// Why patch the getters rather than call the public CommitWithState(), which
/// is the API InControl actually offers for this:
///
///   1. Ordering. Patching intercepts at READ time, so it does not matter
///      whether our Update() runs before or after HeroController's or
///      InputHandler's - Unity does not guarantee an order between them.
///      CommitWithState would have to land in a specific window each frame.
///   2. moveVector. The knight's horizontal movement is
///      `inputActions.moveVector.Vector.x`, NOT `left.IsPressed`. moveVector
///      is a PlayerTwoAxisAction that recomputes itself from left/right via
///      Utility.ValueFromSides(), which takes floats - so it reads its source
///      actions through OneAxisInputControl's implicit float operator, which
///      is the Value getter. Patching Value therefore feeds moveVector for
///      free; committing state to left/right would additionally have to beat
///      PlayerTwoAxisAction.Update() to the frame.
///
/// Pass-through is the default: an action the trainer is NOT holding reports
/// whatever the real hardware says, so you can still pick up a controller and
/// play while the mod is loaded.
/// </summary>
public static class InputInjector
{
    // Logical action name (what Python sends) -> persistent injected state.
    // Names are the trainer's vocabulary, deliberately not the game's field
    // names, so Python never has to care about the in-game key bindings.
    private static readonly Dictionary<string, ActionState> Actions =
        CreateActions();

    // Keyed by the control instance so a patched getter finds its own state in
    // one lookup. Publish a fresh dictionary when HeroActions is replaced;
    // published dictionaries are never mutated, so getters stay lock-free.
    private static volatile Dictionary<OneAxisInputControl, ActionState> States =
        new Dictionary<OneAxisInputControl, ActionState>();

    // Commands arrive on socket threads; Unity state may only be touched on
    // the main thread. This queue is the only thing shared across threads, and
    // the only thing that needs a lock.
    private static readonly Queue<Command> Pending = new Queue<Command>();
    private static readonly object QueueGate = new object();

    private static HeroActions boundActions;
    private static bool patchAttempted;
    private static bool gettersPatched;
    public static bool Ready { get; private set; }

    private sealed class ActionState
    {
        public bool Held;          // trainer wants this down right now
        public bool HeldLastFrame; // for the WasPressed/WasReleased edges
        public int PressedFrame;   // guarantees a tap survives one frame
        public float ReleaseAt;    // unscaled time; 0 = held indefinitely

        // True while we are driving this action. The extra frame after release
        // is what lets WasReleased be observed before we hand the action back
        // to the real hardware.
        public bool Overriding { get { return Held || HeldLastFrame; } }
    }

    private struct Command
    {
        public string Kind;    // "KD" | "KU" | "TAP" | "CLEAR" | "KILL"
        public string Action;
        public float HoldSeconds;
    }

    // ----------------------------------------------------------------------
    // Setup

    /// <summary>
    /// Refresh action bindings on the main thread. The game can replace its
    /// HeroActions during scene/input setup; retain logical held/edge state
    /// while rebinding the new controls. Harmony getters are patched once.
    /// </summary>
    public static bool TryResolve(BepInEx.Logging.ManualLogSource log)
    {
        InputHandler handler = InputHandler.Instance;
        if (handler == null || handler.inputActions == null)
        {
            Ready = false;
            if (boundActions != null)
            {
                States = new Dictionary<OneAxisInputControl, ActionState>();
                boundActions = null;
            }
            return false;
        }

        HeroActions a = handler.inputActions;
        if (object.ReferenceEquals(a, boundActions))
            return Ready;

        var bindings = new Dictionary<OneAxisInputControl, ActionState>();
        bool complete = Bind(bindings, "left", a.left);
        complete &= Bind(bindings, "right", a.right);
        complete &= Bind(bindings, "up", a.up);
        complete &= Bind(bindings, "down", a.down);
        complete &= Bind(bindings, "jump", a.jump);
        complete &= Bind(bindings, "attack", a.attack);
        complete &= Bind(bindings, "dash", a.dash);
        complete &= Bind(bindings, "cast", a.cast);
        complete &= Bind(bindings, "quick_cast", a.quickCast);
        complete &= Bind(bindings, "super_dash", a.superDash);
        complete &= Bind(bindings, "dream_nail", a.dreamNail);
        complete &= Bind(bindings, "submit", a.menuSubmit);
        complete &= Bind(bindings, "cancel", a.menuCancel);
        complete &= Bind(bindings, "pause", a.pause);
        if (!complete)
        {
            Ready = false;
            States = new Dictionary<OneAxisInputControl, ActionState>();
            boundActions = null;
            return false;
        }

        States = bindings;
        boundActions = a;
        if (!patchAttempted)
        {
            patchAttempted = true;
            gettersPatched = PatchGetters(log);
        }
        Ready = gettersPatched;
        log.LogInfo("HollowRL: refreshed input action bindings (" + bindings.Count + " actions)");
        return Ready;
    }

    private static Dictionary<string, ActionState> CreateActions()
    {
        var actions = new Dictionary<string, ActionState>();
        foreach (string name in new[] {
            "left", "right", "up", "down", "jump", "attack", "dash",
            "cast", "quick_cast", "super_dash", "dream_nail", "submit", "cancel", "pause",
        })
            actions.Add(name, new ActionState());
        return actions;
    }

    private static bool Bind(Dictionary<OneAxisInputControl, ActionState> bindings,
                             string name, PlayerAction action)
    {
        if (action == null)
            return false;
        bindings[action] = Actions[name];
        return true;
    }

    /// <summary>
    /// Patch the five getters the game reads actions through. All of them are
    /// declared on OneAxisInputControl, not on PlayerAction - patching the
    /// base is why the dictionary lookup below has to filter by instance.
    /// </summary>
    private static bool PatchGetters(BepInEx.Logging.ManualLogSource log)
    {
        var harmony = new Harmony("com.AbdulAzizOM.HollowRL.input");

        var getters = new Dictionary<string, string>
        {
            { "IsPressed",   nameof(IsPressedPrefix) },
            { "WasPressed",  nameof(WasPressedPrefix) },
            { "WasReleased", nameof(WasReleasedPrefix) },
            { "State",       nameof(IsPressedPrefix) },
            { "Value",       nameof(ValuePrefix) },
            { "RawValue",    nameof(ValuePrefix) },
        };

        int patched = 0;

        foreach (var entry in getters)
        {
            MethodInfo getter = AccessTools.PropertyGetter(
                typeof(OneAxisInputControl), entry.Key);

            if (getter == null)
            {
                log.LogWarning("HollowRL: no getter for " + entry.Key + " - skipped");
                continue;
            }

            MethodInfo prefix = typeof(InputInjector).GetMethod(
                entry.Value, BindingFlags.NonPublic | BindingFlags.Static);

            harmony.Patch(getter, prefix: new HarmonyMethod(prefix));
            patched++;
        }

        if (patched == 0)
        {
            log.LogError("HollowRL: patched no input getters - injection is DISABLED");
            return false;
        }

        log.LogInfo("HollowRL: input injection active (" + patched
                    + " getters patched, " + Actions.Count + " actions bound)");
        return true;
    }

    // ----------------------------------------------------------------------
    // The patches.
    //
    // These run on EVERY read of EVERY OneAxisInputControl in the game - menus
    // and device controls included - so they are deliberately lock-free:
    // States always refers to an immutable binding snapshot. ActionState
    // fields are written only by Tick(), on the main thread that reads them.
    //
    // Returning false skips the original getter and uses our __result.

    /// <summary>
    /// Our state, or null if this control is not one we are currently driving.
    ///
    /// EnabledInHierarchy is the real getters' own gate - they all return
    /// false/0 when the action set is disabled, which the game does during
    /// menus, cutscenes and pauses. Deferring to it means injected input is
    /// suppressed exactly where a real key press would be, instead of the
    /// agent being the one thing in the game that can act during a cutscene.
    /// </summary>
    private static ActionState Driving(OneAxisInputControl control)
    {
        ActionState state;
        if (!States.TryGetValue(control, out state))
            return null;
        if (!state.Overriding || !control.EnabledInHierarchy)
            return null;
        return state;
    }

    private static bool IsPressedPrefix(OneAxisInputControl __instance, ref bool __result)
    {
        ActionState state = Driving(__instance);
        if (state == null)
            return true;   // not ours - the real hardware answers

        __result = state.Held;
        return false;
    }

    private static bool WasPressedPrefix(OneAxisInputControl __instance, ref bool __result)
    {
        ActionState state = Driving(__instance);
        if (state == null)
            return true;

        __result = state.Held && !state.HeldLastFrame;
        return false;
    }

    private static bool WasReleasedPrefix(OneAxisInputControl __instance, ref bool __result)
    {
        ActionState state = Driving(__instance);
        if (state == null)
            return true;

        __result = !state.Held && state.HeldLastFrame;
        return false;
    }

    private static bool ValuePrefix(OneAxisInputControl __instance, ref float __result)
    {
        ActionState state = Driving(__instance);
        if (state == null)
            return true;

        __result = state.Held ? 1f : 0f;
        return false;
    }

    // ----------------------------------------------------------------------
    // Commands (called from socket threads - queue only, no Unity API)

    public static void Enqueue(string kind, string action, float holdSeconds)
    {
        lock (QueueGate)
        {
            Pending.Enqueue(new Command
            {
                Kind = kind,
                Action = action,
                HoldSeconds = holdSeconds,
            });
        }
    }

    public static bool IsKnownAction(string action)
    {
        return Actions.ContainsKey(action);
    }

    /// <summary>Release everything. Called when a client disconnects, so a
    /// crashed trainer cannot leave the knight running into a wall for the
    /// rest of the session.</summary>
    public static void ReleaseAll()
    {
        Enqueue("CLEAR", null, 0f);
    }

    // ----------------------------------------------------------------------
    // Per-frame application (main thread)

    /// <summary>
    /// Advance the edge state, apply queued commands, and expire timed taps.
    /// Must run once per frame. Returns true if a KILL was requested since the
    /// last frame.
    ///
    /// No Ready guard: logical state and CLEAR/tap expiry must still advance
    /// while InputHandler is absent, and KILL must still reach episode resets.
    /// </summary>
    public static bool Tick()
    {
        bool kill = false;
        float now = Time.unscaledTime;
        int frame = Time.frameCount;

        foreach (ActionState state in Actions.Values)
            state.HeldLastFrame = state.Held;

        // Drain into a local array so the lock is held only for the dequeue,
        // never while touching action state.
        Command[] batch = null;
        lock (QueueGate)
        {
            if (Pending.Count > 0)
            {
                batch = Pending.ToArray();
                Pending.Clear();
            }
        }

        if (batch != null)
        {
            foreach (Command cmd in batch)
            {
                if (cmd.Kind == "KILL")
                {
                    kill = true;
                    continue;
                }

                if (cmd.Kind == "CLEAR")
                {
                    foreach (ActionState state in Actions.Values)
                    {
                        state.Held = false;
                        state.ReleaseAt = 0f;
                    }
                    continue;
                }

                ActionState target;
                if (!Actions.TryGetValue(cmd.Action, out target))
                    continue;

                switch (cmd.Kind)
                {
                    case "KD":
                        if (!target.Held)
                            target.PressedFrame = frame;
                        target.Held = true;
                        target.ReleaseAt = 0f;
                        break;

                    case "KU":
                        target.Held = false;
                        target.ReleaseAt = 0f;
                        break;

                    case "TAP":
                        if (!target.Held)
                            target.PressedFrame = frame;
                        target.Held = true;
                        target.ReleaseAt = now + cmd.HoldSeconds;
                        break;
                }
            }
        }

        // Expire timed taps. The frame check is what makes a short tap work at
        // all: a 60 ms hold requested and expired within the same frame would
        // be pressed and released before the game ever polled it, and simply
        // never happen. This is the same problem pyautogui.press() had, which
        // is why the Python side grew an explicit hold duration.
        foreach (ActionState state in Actions.Values)
        {
            if (state.Held && state.ReleaseAt > 0f
                && now >= state.ReleaseAt
                && frame > state.PressedFrame)
            {
                state.Held = false;
                state.ReleaseAt = 0f;
            }
        }

        return kill;
    }
}
