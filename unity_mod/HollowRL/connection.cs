using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text;
using System.Threading;
using BepInEx;
using HarmonyLib;
using UnityEngine;
using Bounds = UnityEngine.Bounds;



[BepInPlugin("com.AbdulAzizOM.HollowRL", "HollowRL", "1.0.0")]
public partial class HollowRL : BaseUnityPlugin
{
    private TcpListener server;
    private Thread serverThread;

    private KnightPos.DiagnosticsOverlay knightPos;
    private readonly object posLock = new object();
    private bool knightPresent;
    private Bounds knightBox;
    private bool attacking;
    private Bounds? attackBox;
    private List<KnightPos.HazardSnapshot> hazards = new List<KnightPos.HazardSnapshot>();
    private int hitCount;
    private int damageCount;
    private Vector2 knightVelocity;
    private bool knightFacingRight;
    private int bossHealth;
    private int bossFacing;
    private Vector2 bossVelocity;
    private Vector2 bossPosition;
    private Bounds? bossBox;
    private string bossControlState = "";
    private int knightHealth;
    private int mana;

    private bool wasInvulnerable;
    private bool bossHealthTracked;
    private int trackedBossId;
    private int trackedBossHealth;
    private bool knightHealthTracked;
    private int trackedKnightHealth;
    // Boss hp reached <=0 at least once since the last reset. Persists while
    // the boss object is gone (death anim / scene teardown) so a "defeated"
    // state is not reported as "missing" and vice versa.
    private bool bossDefeated = false;
    private bool fightActive;
    private bool knightDiedThisFight;
    private int fightBossId = int.MinValue;

    private bool isDashing;
    private bool isJumping;
    private bool isFalling;
    private bool isOnGround;
    private bool isRecoiling;
    private bool isTouchingWallLeft;
    private bool isTouchingWallRight;
    private float wallDistanceLeft = 100f;
    private float wallDistanceRight = 100f;
    private bool canShadeCloakNow;
    private float gameSpeed = 1f;
    private bool trainingPaused;
    private volatile float requestedGameSpeed = -1f;
    private volatile bool serverRunning = true;
    // Boss Control FSM name. Pure Vessel uses "Control"; Traitor Lord uses
    // "Mantis". Overridable via --hollowrl-boss-fsm= without a recompile.
    private readonly string bossFsmName = ResolveBossFsmName();

    private static string ResolveBossFsmName()
    {
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            const string prefix = "--hollowrl-boss-fsm=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return arg.Substring(prefix.Length);
        }
        return "Control";
    }

    private int ResolvePort()
    {
        const int fallback = 9000;
        string raw = Environment.GetEnvironmentVariable("HOLLOWRL_PORT");
        int port;
        if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out port)
                && port >= 1024 && port <= 65535)
            return port;

        foreach (string arg in Environment.GetCommandLineArgs())
        {
            const string prefix = "--hollowrl-port=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                    && int.TryParse(arg.Substring(prefix.Length), out port)
                    && port >= 1024 && port <= 65535)
                return port;
        }

        return fallback;
    }


    private static string ResolveBindIp()
    {
        // Default to loopback so a single-machine setup behaves exactly as
        // before. Remote workers pass --hollowrl-bind-ip=HOST to listen on a
        // specific address; never fall back to IPAddress.Any.
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            const string prefix = "--hollowrl-bind-ip=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return arg.Substring(prefix.Length);
        }
        return "127.0.0.1";
    }


    private static int ResolveFrameRate()
    {
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            const string prefix = "--hollowrl-fps=";
            int value;
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                    && int.TryParse(arg.Substring(prefix.Length), out value)
                    && value >= 30 && value <= 240)
                return value;
        }
        return 0;
    }


    private static float ResolveGameSpeed()
    {
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            const string prefix = "--hollowrl-speed=";
            float value;
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                    && float.TryParse(arg.Substring(prefix.Length),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture,
                        out value)
                    && value >= 0.5f && value <= 4f)
                return value;
        }
        return -1f;
    }

    private void Awake()
    {
        resetPlugin = this;
        Harmony.CreateAndPatchAll(typeof(HollowRL));
        Harmony.CreateAndPatchAll(typeof(TrainingDeathPatch), "com.AbdulAzizOM.HollowRL.reset");

        // Multiple windowed instances must keep simulating while another one
        // has focus.  An optional frame cap saves a large amount of rendering
        // work without making the 60 ms input taps shorter than one frame.
        Application.runInBackground = true;
        int frameRate = ResolveFrameRate();
        if (frameRate > 0)
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = frameRate;
        }
        float startupSpeed = ResolveGameSpeed();
        if (startupSpeed > 0f)
            Time.timeScale = startupSpeed;

        knightPos = gameObject.AddComponent<KnightPos.DiagnosticsOverlay>();

        int port = ResolvePort();
        string bindIp = ResolveBindIp();
        IPAddress bindAddress;
        try
        {
            bindAddress = IPAddress.Parse(bindIp);
        }
        catch (Exception exc)
        {
            Logger.LogError("HollowRL: invalid --hollowrl-bind-ip '" + bindIp + "': " + exc.Message);
            return;
        }
        server = new TcpListener(bindAddress, port);
        server.Start();

        serverThread = new Thread(ServerLoop);
        serverThread.IsBackground = true;
        serverThread.Start();

        Logger.LogInfo("TCP server started on " + bindAddress + ":" + port);
    }


    private void Update()
    {
        FightResetTick();
        // Input injection has to be wired up before anything reads input this
        // frame. Scene/input setup can replace HeroActions after the first
        // binding, so check its identity every frame and rebind as needed.
        InputInjector.TryResolve(Logger);

        // Apply whatever the trainer asked for since the last frame. Commands
        // arrive on socket threads and are only ever applied here, on the main
        // thread, so the game sees a consistent input state for the frame.
        bool killRequested = InputInjector.Tick();

        // O: toggle 2x game speed. P: toggle training pause. Read here (main
        // thread, every frame) and applied/stored below under the lock so the
        // socket thread sees a consistent snapshot. These stay real keyboard
        // hotkeys - they are for the human watching, not for the agent.
        float speedRequest = requestedGameSpeed;
        if (speedRequest >= 0f)
        {
            requestedGameSpeed = -1f;
            Time.timeScale = speedRequest;
        }

        bool speedToggle = Input.GetKeyDown(KeyCode.O);
        bool pauseToggle = Input.GetKeyDown(KeyCode.P);

        if (speedToggle)
            Time.timeScale = Mathf.Approximately(Time.timeScale, 1f) ? 2f : 1f;

        lock (posLock)
        {
            gameSpeed = Time.timeScale;
            if (pauseToggle)
                trainingPaused = !trainingPaused;
        }

        // Instant death cheat, for forcing episode resets during training.
        // Y still works for the human; the trainer now asks over the socket
        // instead of firing a desktop-wide keystroke and hoping the game had
        // focus.
        if ((killRequested || Input.GetKeyDown(KeyCode.Y))
                && HeroController.instance != null && PlayerData.instance != null)
            KillKnight();

        // HeroController does not exist in the title screen or during scene
        // transitions. Previously the first read below threw every frame,
        // filling both Unity's and BepInEx's logs with NullReferenceException
        // until a save finished loading. Publish an explicit "no knight"
        // snapshot instead; Python treats it as an unavailable observation.
        if (HeroController.instance == null || PlayerData.instance == null)
        {
            bool victoryDetectedWithoutHero = false;
            if (fightActive)
            {
                if (trackedKnightHealth <= 0)
                {
                    knightDiedThisFight = true;
                    bossDefeated = false;
                }
                else
                {
                    // Some victory transitions remove both scene objects in
                    // one frame. The last main-thread snapshot still tells us
                    // the knight was alive when the active fight vanished.
                    bossDefeated = true;
                    victoryDetectedWithoutHero = true;
                }
                fightActive = false;
            }
            lock (posLock)
            {
                knightPresent = false;
                attacking = false;
                attackBox = null;
                hazards = new List<KnightPos.HazardSnapshot>();
                bossHealth = bossDefeated ? 0 : -1;
                bossFacing = -1;
                bossVelocity = Vector2.zero;
                bossPosition = Vector2.zero;
                bossBox = null;
                bossControlState = "";
                knightVelocity = Vector2.zero;
                knightHealth = 0;
                mana = 0;
                isDashing = false;
                isJumping = false;
                isFalling = false;
                isOnGround = false;
                isRecoiling = false;
                isTouchingWallLeft = false;
                isTouchingWallRight = false;
                wallDistanceLeft = 100f;
                wallDistanceRight = 100f;
                canShadeCloakNow = false;
                wasInvulnerable = false;
                bossHealthTracked = false;
                knightHealthTracked = false;
                if (victoryDetectedWithoutHero)
                    hitCount++;
            }
            return;
        }

        Bounds box = knightPos.GetKnightBox();
        bool isAttacking = knightPos.IsAttacking();
        Bounds? atkBox = knightPos.GetAttackBox();
        bool invulnerable = knightPos.IsInvulnerable();
        Vector2 velocity = knightPos.GetKnightVelocity();
        bool facingRight = knightPos.IsKnightFacingRight();
        // One scene lookup instead of three - see the comment on
        // GetBossHealth() in KnightPos.cs.
        HealthManager boss = knightPos.GetBoss();
        int? bossHp = knightPos.GetBossHealth(boss);
        bool? bossFacingRight = knightPos.IsBossFacingRight(boss);
        Vector2 bVelocity = knightPos.GetBossVelocity(boss);
        Vector2 bPosition = knightPos.GetBossPosition(boss);
        Collider2D bCollider = knightPos.GetBossCollider(boss);
        Bounds? bBox = bCollider != null ? bCollider.bounds : (Bounds?)null;
        List<KnightPos.HazardSnapshot> currentHazards = knightPos.GetHazards(boss);
        string controlState = ReadControlFsmState(boss);
        int kHealth = knightPos.GetKnightHealth();
        int manaAmount = knightPos.GetMana();
        CaptureFightReset(boss, kHealth, manaAmount);

        int health;
        bool victoryDetectedThisFrame = false;
        if (bossHp.HasValue)
        {
            health = bossHp.Value;
            int bossId = boss.GetInstanceID();
            if (health > 0)
            {
                if (!fightActive || fightBossId != bossId)
                {
                    fightActive = true;
                    fightBossId = bossId;
                    knightDiedThisFight = false;
                    bossDefeated = false;
                }
                if (kHealth <= 0)
                    knightDiedThisFight = true;
            }
            else
            {
                bossDefeated = true;
                fightActive = false;
            }
        }
        else
        {
            // Pure Vessel disables/destroys its HealthManager immediately on
            // the killing blow, so the last readable value can be 16 rather
            // than 0. If a live fight loses its boss while the knight is still
            // alive, that is a confirmed victory. A knight death is observed
            // on the main Unity thread first and must never take this branch.
            if (fightActive)
            {
                if (kHealth <= 0)
                    knightDiedThisFight = true;
                if (!knightDiedThisFight && kHealth > 0)
                {
                    bossDefeated = true;
                    victoryDetectedThisFrame = true;
                }
                fightActive = false;
            }
            health = bossDefeated ? 0 : -1;
        }

        bool dashing = knightPos.IsDashing();
        bool jumping = knightPos.IsJumping();
        bool falling = knightPos.IsFalling();
        bool onGround = knightPos.IsOnGround();
        bool recoiling = knightPos.IsRecoiling();
        bool touchingWallLeft = knightPos.IsTouchingWallLeft();
        bool touchingWallRight = knightPos.IsTouchingWallRight();
        Vector2 wallDistances = knightPos.GetWallDistances();
        bool shadeCloakAvailable = knightPos.CanShadeCloakNow();

        int newHits = 0;
        if (boss != null)
        {
            int bossId = boss.GetInstanceID();
            if (bossHealthTracked && trackedBossId == bossId && health < trackedBossHealth)
                newHits = 1;

            bossHealthTracked = true;
            trackedBossId = bossId;
            trackedBossHealth = health;
        }
        else
        {
            if (victoryDetectedThisFrame && bossHealthTracked)
                newHits = 1;
            bossHealthTracked = false;
        }

        int newDamage = 0;
        if (knightHealthTracked && kHealth < trackedKnightHealth)
            newDamage = trackedKnightHealth - kHealth;
        knightHealthTracked = true;
        trackedKnightHealth = kHealth;

        lock (posLock)
        {
            knightPresent = true;
            knightBox = box;
            attacking = isAttacking;
            attackBox = atkBox;
            hazards = currentHazards;
            knightVelocity = velocity;
            knightFacingRight = facingRight;
            bossHealth = health;
            bossFacing = bossFacingRight.HasValue ? (bossFacingRight.Value ? 1 : 0) : -1;
            bossVelocity = bVelocity;
            bossPosition = bPosition;
            bossBox = bBox;
            bossControlState = controlState;
            knightHealth = kHealth;
            mana = manaAmount;

            isDashing = dashing;
            isJumping = jumping;
            isFalling = falling;
            isOnGround = onGround;
            isRecoiling = recoiling;
            isTouchingWallLeft = touchingWallLeft;
            isTouchingWallRight = touchingWallRight;
            wallDistanceLeft = wallDistances.x;
            wallDistanceRight = wallDistances.y;
            canShadeCloakNow = shadeCloakAvailable;

            hitCount += newHits;
            damageCount += newDamage;
            wasInvulnerable = invulnerable;
        }

    }


    private string ReadControlFsmState(HealthManager boss)
    {
        if (boss == null)
            return "";

        Component[] components = boss.GetComponentsInChildren<Component>(true);
        foreach (Component component in components)
        {
            if (component == null || component.GetType().Name != "PlayMakerFSM")
                continue;

            object fsmName = GetMemberValue(component, "FsmName");
            if ((fsmName as string) != bossFsmName)
                continue;

            object activeState = GetMemberValue(component, "ActiveStateName");
            if (activeState == null)
            {
                object fsm = GetMemberValue(component, "Fsm");
                activeState = GetMemberValue(fsm, "ActiveStateName");
            }
            return Sanitize(activeState);
        }
        return "";
    }

    private static object GetMemberValue(object target, string name)
    {
        if (target == null)
            return null;

        Type type = target.GetType();
        const System.Reflection.BindingFlags flags =
            System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.NonPublic;

        System.Reflection.PropertyInfo property = type.GetProperty(name, flags);
        if (property != null && property.GetIndexParameters().Length == 0)
            return property.GetValue(target, null);

        System.Reflection.FieldInfo field = type.GetField(name, flags);
        return field != null ? field.GetValue(target) : null;
    }


    private static string FormatFloat(float value)
    {
        return value.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
    }


    private static string Sanitize(object value)
    {
        if (value == null)
            return "";
        return value.ToString()
            .Replace("\t", " ")
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Replace("~", "-")
            .Replace("|", "/");
    }


    /// <summary>
    /// Instant death, bypassing i-frames. Die() only checks cState.dead (to
    /// avoid double-triggering the death sequence), not invulnerability, so
    /// zeroing health and invoking it directly works where TakeDamage would
    /// be silently swallowed by CanTakeDamage() mid-invuln.
    /// </summary>
    private void KillKnight()
    {
        PlayerData.instance.health = 0;
        IEnumerator dieRoutine = Traverse.Create(HeroController.instance).Method("Die").GetValue<IEnumerator>();
        HeroController.instance.StartCoroutine(dieRoutine);
    }


    private void ServerLoop()
    {
        // AcceptTcpClient() used to be called exactly once, outside any loop,
        // so the mod served ONE python client per game session. After that
        // client went away the loop exited and nothing ever accepted again -
        // but the TcpListener stayed bound, so later connects still succeeded
        // via the OS backlog and then hung with no reply. Restarting main_multi.py
        // (or switching between it and the dashboard) bricked the mod until
        // the game itself was restarted. Accept in a loop instead.
        //
        // Each client now gets its own thread. Accepting sequentially served
        // exactly one at a time, which is no longer enough: the trainer keeps
        // a second connection open for input commands (see controls.py) on
        // top of the state poll, and the dashboard is often a third.
        while (serverRunning)
        {
            try
            {
                TcpClient client = server.AcceptTcpClient();

                Thread worker = new Thread(() => HandleClient(client));
                worker.IsBackground = true;
                worker.Start();
            }
            catch (Exception exc)
            {
                // server.Stop() from OnDestroy makes the pending Accept throw;
                // that is a shutdown, not an error.
                if (!serverRunning)
                    return;

                Logger.LogWarning("HollowRL: accept failed - " + exc.Message);
            }
        }
    }


    private void HandleClient(TcpClient client)
    {
        try
        {
            client.NoDelay = true;
            StreamReader reader = new StreamReader(client.GetStream());
            StreamWriter writer = new StreamWriter(client.GetStream());

            writer.AutoFlush = true;

            Logger.LogInfo("HollowRL: client connected");

            while (client.Connected)
            {
                string command = reader.ReadLine();

                // ReadLine() returns null as soon as the peer closes.
                // There was no break here, so a disconnected client left
                // this spinning on a dead socket - client.Connected stays
                // true until the next failed write - pegging a CPU core
                // inside the game process, of all places.
                if (command == null)
                    break;

                if (HandleDemonstrationCommand(client, command, writer))
                    continue;

                string resetReply = HandleFightResetCommand(client, command);
                if (resetReply != null)
                {
                    writer.WriteLine(resetReply);
                    continue;
                }

                // Anything that is not a state request is an input command.
                if (!command.StartsWith("pos"))
                {
                    writer.WriteLine(HandleInputCommand(command));
                    continue;
                }

                writer.WriteLine(BuildStateLine());
            }

            Logger.LogInfo("HollowRL: client disconnected");
        }
        catch (Exception exc)
        {
            if (serverRunning)
                Logger.LogWarning("HollowRL: client dropped - " + exc.Message);
        }
        finally
        {
            ReleaseFightResetOwner(client);
            StopDemonstrationOnDisconnect(client);
            // A trainer that crashes mid-episode must not leave the knight
            // holding `left` for the rest of the session.
            InputInjector.ReleaseAll();
            client.Close();
        }
    }


    /// <summary>
    /// Parse one input command. Runs on a socket thread, so it only ever
    /// queues - InputInjector.Tick() applies it on the main thread.
    ///
    ///   KD &lt;action&gt;         hold
    ///   KU &lt;action&gt;         release
    ///   TAP &lt;action&gt; &lt;ms&gt;   hold for ms, released by the mod
    ///   CLEAR                     release everything
    ///   KILL                      instant death, to force an episode reset
    ///   SPEED &lt;factor&gt;            set Unity timeScale on the main thread
    /// </summary>
    private string HandleInputCommand(string command)
    {
        string[] parts = command.Trim().Split(' ');
        string verb = parts[0].ToUpperInvariant();

        if (DemonstrationActive)
            return "err human demonstration active; stop the recorder before injecting controls";

        if (verb == "CLEAR" || verb == "KILL")
        {
            InputInjector.Enqueue(verb, null, 0f);
            return "ok";
        }

        if (verb == "SPEED")
        {
            float factor;
            if (parts.Length < 2 || !float.TryParse(parts[1],
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out factor) || factor < 0f || factor > 4f)
                return "err bad speed";
            requestedGameSpeed = factor;
            return "ok";
        }

        if (verb != "KD" && verb != "KU" && verb != "TAP")
            return "err unknown command";

        if (parts.Length < 2)
            return "err missing action";

        string action = parts[1];
        if (!InputInjector.IsKnownAction(action))
            return "err unknown action " + action;

        float holdSeconds = 0f;
        if (verb == "TAP")
        {
            float ms;
            if (parts.Length < 3 || !float.TryParse(parts[2],
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out ms))
                return "err bad duration";
            holdSeconds = ms / 1000f;
        }

        InputInjector.Enqueue(verb, action, holdSeconds);
        return "ok";
    }


    /// <summary>
    /// One frame of state, copied out of the fields Update() writes. Exists so
    /// the lock is held for exactly one struct copy and the formatting below
    /// happens outside it. Mirrors game_state.Frame on the Python side.
    /// </summary>
    private struct StateSnapshot
    {
        public bool KnightPresent;
        public Bounds Knight;
        public bool Attacking;
        public Bounds? AttackBox;
        public List<KnightPos.HazardSnapshot> Hazards;
        public int Hits;
        public int Damages;
        public Vector2 KnightVelocity;
        public bool KnightFacingRight;
        public int BossHealth;
        public int BossFacing;
        public Vector2 BossVelocity;
        public Vector2 BossPosition;
        public Bounds? BossBox;
        public string ControlFsmState;
        public int KnightHealth;
        public int Mana;
        public float GameSpeed;
        public bool Dashing;
        public bool Jumping;
        public bool Falling;
        public bool OnGround;
        public bool Recoiling;
        public bool TouchingWallLeft;
        public bool TouchingWallRight;
        public float WallDistanceLeft;
        public float WallDistanceRight;
        public bool ShadeCloakAvailable;
        public bool Invulnerable;
        public bool Paused;
    }


    private StateSnapshot TakeSnapshot()
    {
        lock (posLock)
        {
            return new StateSnapshot
            {
                KnightPresent = knightPresent,
                Knight = knightBox,
                Attacking = attacking,
                AttackBox = attackBox,
                Hazards = hazards,
                Hits = hitCount,
                Damages = damageCount,
                KnightVelocity = knightVelocity,
                KnightFacingRight = knightFacingRight,
                BossHealth = bossHealth,
                BossFacing = bossFacing,
                BossVelocity = bossVelocity,
                BossPosition = bossPosition,
                BossBox = bossBox,
                ControlFsmState = bossControlState,
                KnightHealth = knightHealth,
                Mana = mana,
                GameSpeed = gameSpeed,
                Dashing = isDashing,
                Jumping = isJumping,
                Falling = isFalling,
                OnGround = isOnGround,
                Recoiling = isRecoiling,
                TouchingWallLeft = isTouchingWallLeft,
                TouchingWallRight = isTouchingWallRight,
                WallDistanceLeft = wallDistanceLeft,
                WallDistanceRight = wallDistanceRight,
                ShadeCloakAvailable = canShadeCloakNow,
                Invulnerable = wasInvulnerable,
                Paused = trainingPaused,
            };
        }
    }


    /// <summary>Append a box as min_x,min_y,max_x,max_y (no separator).</summary>
    private static void AppendBox(StringBuilder sb, Bounds box)
    {
        sb.Append(box.min.x).Append(',')
          .Append(box.min.y).Append(',')
          .Append(box.max.x).Append(',')
          .Append(box.max.y);
    }


    /// <summary>
    /// Serialise one frame for the `pos` command. Sections are separated by
    /// ';', fields within a section by ','. Parsed by _parse_frame() in
    /// game_state.py - the two must be changed together.
    ///
    ///   0:  knight box       min_x,min_y,max_x,max_y
    ///   1:  attacking        1/0
    ///   2:  attack box       as section 0, or empty when not swinging
    ///   3:  counters         hits,damages  (lifetime, never reset)
    ///   4:  scalars          vel_x,vel_y,facing_right,boss_health,boss_facing,
    ///                        boss_vel_x,boss_vel_y,knight_health,mana,speed,
    ///                        boss_x,boss_y,wall_distance_left,wall_distance_right
    ///   5:  flags            dashing,jumping,falling,recoiling,wall_left,
    ///                        wall_right,shade_cloak,invuln,paused,on_ground
    ///   6:  control fsm      PlayMaker "Control" FSM ActiveStateName, or empty
    ///   7:  boss box         min_x,min_y,max_x,max_y, or empty
    ///   8+: one hazard       id,min_x,min_y,max_x,max_y,vel_x,vel_y,boss_attached
    /// </summary>
    private string BuildStateLine()
    {
        return BuildStateLine(TakeSnapshot());
    }


    private string BuildStateLine(StateSnapshot s)
    {

        var sb = new StringBuilder();
        if (s.KnightPresent)
            AppendBox(sb, s.Knight);

        sb.Append(';').Append(s.Attacking ? 1 : 0);

        sb.Append(';');
        if (s.AttackBox.HasValue)
            AppendBox(sb, s.AttackBox.Value);

        sb.Append(';').Append(s.Hits).Append(',').Append(s.Damages);

        sb.Append(';').Append(s.KnightVelocity.x).Append(',')
          .Append(s.KnightVelocity.y).Append(',')
          .Append(s.KnightFacingRight ? 1 : 0).Append(',')
          .Append(s.BossHealth).Append(',')
          .Append(s.BossFacing).Append(',')
          .Append(s.BossVelocity.x).Append(',')
          .Append(s.BossVelocity.y).Append(',')
          .Append(s.KnightHealth).Append(',')
          .Append(s.Mana).Append(',')
          .Append(s.GameSpeed).Append(',')
          .Append(s.BossPosition.x).Append(',')
          .Append(s.BossPosition.y).Append(',')
          .Append(s.WallDistanceLeft).Append(',')
          .Append(s.WallDistanceRight);

        sb.Append(';')
          .Append(s.Dashing ? 1 : 0).Append(',')
          .Append(s.Jumping ? 1 : 0).Append(',')
          .Append(s.Falling ? 1 : 0).Append(',')
          .Append(s.Recoiling ? 1 : 0).Append(',')
          .Append(s.TouchingWallLeft ? 1 : 0).Append(',')
          .Append(s.TouchingWallRight ? 1 : 0).Append(',')
          .Append(s.ShadeCloakAvailable ? 1 : 0).Append(',')
          .Append(s.Invulnerable ? 1 : 0).Append(',')
          .Append(s.Paused ? 1 : 0).Append(',')
          .Append(s.OnGround ? 1 : 0);

        sb.Append(';').Append(s.ControlFsmState);
        sb.Append(';');
        if (s.BossBox.HasValue)
            AppendBox(sb, s.BossBox.Value);

        foreach (KnightPos.HazardSnapshot hazard in s.Hazards)
        {
            Bounds box = hazard.Box;
            sb.Append(';').Append(hazard.InstanceId).Append(',')
              .Append(box.min.x).Append(',')
              .Append(box.min.y).Append(',')
              .Append(box.max.x).Append(',')
              .Append(box.max.y).Append(',')
              .Append(hazard.Velocity.x).Append(',')
              .Append(hazard.Velocity.y).Append(',')
              .Append(hazard.BossAttached ? 1 : 0);
        }

        return sb.ToString();
    }


    private void OnDestroy()
    {
        if (resetPlugin == this) resetPlugin = null;
        serverRunning = false;

        if (server != null)
            server.Stop();
    }


}
