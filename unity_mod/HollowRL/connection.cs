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
public class HollowRL : BaseUnityPlugin
{
    private TcpListener server;
    private Thread serverThread;

    private KnightPos.DiagnosticsOverlay knightPos;
    private readonly object posLock = new object();
    private bool knightPresent;
    private Bounds knightBox;
    private bool attacking;
    private Bounds? attackBox;
    private List<Bounds> hazardBoxes = new List<Bounds>();
    private int hitCount;
    private int damageCount;
    private Vector2 knightVelocity;
    private bool knightFacingRight;
    private int bossHealth;
    private int bossFacing;
    private Vector2 bossVelocity;
    private int knightHealth;
    private int mana;

    private bool wasHitting;
    private bool wasDamaged;
    private bool wasInvulnerable;

    private bool isDashing;
    private bool isJumping;
    private bool isFalling;
    private bool isRecoiling;
    private bool isTouchingWallLeft;
    private bool isTouchingWallRight;
    private bool canShadeCloakNow;
    private float gameSpeed = 1f;
    private bool trainingPaused;
    private volatile bool serverRunning = true;
    private bool inputReady;

    private void Awake()
    {
        Harmony.CreateAndPatchAll(typeof(HollowRL));

        knightPos = gameObject.AddComponent<KnightPos.DiagnosticsOverlay>();

        server = new TcpListener(IPAddress.Loopback, 9000);
        server.Start();

        serverThread = new Thread(ServerLoop);
        serverThread.IsBackground = true;
        serverThread.Start();

        Logger.LogInfo("TCP server started");
    }


    private void Update()
    {
        // Input injection has to be wired up before anything reads input this
        // frame. InputHandler does not exist until a save is loaded, so this
        // keeps retrying until it resolves and then costs one bool check.
        if (!inputReady)
            inputReady = InputInjector.TryResolve(Logger);

        // Apply whatever the trainer asked for since the last frame. Commands
        // arrive on socket threads and are only ever applied here, on the main
        // thread, so the game sees a consistent input state for the frame.
        bool killRequested = InputInjector.Tick();

        // O: toggle 2x game speed. P: toggle training pause. Read here (main
        // thread, every frame) and applied/stored below under the lock so the
        // socket thread sees a consistent snapshot. These stay real keyboard
        // hotkeys - they are for the human watching, not for the agent.
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
            lock (posLock)
            {
                knightPresent = false;
                attacking = false;
                attackBox = null;
                hazardBoxes = new List<Bounds>();
                bossHealth = 0;
                bossFacing = -1;
                bossVelocity = Vector2.zero;
                knightVelocity = Vector2.zero;
                knightHealth = 0;
                mana = 0;
                isDashing = false;
                isJumping = false;
                isFalling = false;
                isRecoiling = false;
                isTouchingWallLeft = false;
                isTouchingWallRight = false;
                canShadeCloakNow = false;
                wasHitting = false;
                wasDamaged = false;
                wasInvulnerable = false;
            }
            return;
        }

        Bounds box = knightPos.GetKnightBox();
        bool isAttacking = knightPos.IsAttacking();
        Bounds? atkBox = knightPos.GetAttackBox();
        List<Bounds> boxes = knightPos.GetHazardBoxes();
        bool invulnerable = knightPos.IsInvulnerable();
        Vector2 velocity = knightPos.GetKnightVelocity();
        bool facingRight = knightPos.IsKnightFacingRight();
        // One scene lookup instead of three - see the comment on
        // GetBossHealth() in KnightPos.cs.
        HealthManager boss = knightPos.GetBoss();
        int health = knightPos.GetBossHealth(boss);
        bool? bossFacingRight = knightPos.IsBossFacingRight(boss);
        Vector2 bVelocity = knightPos.GetBossVelocity(boss);
        int kHealth = knightPos.GetKnightHealth();
        int manaAmount = knightPos.GetMana();

        bool dashing = knightPos.IsDashing();
        bool jumping = knightPos.IsJumping();
        bool falling = knightPos.IsFalling();
        bool recoiling = knightPos.IsRecoiling();
        bool touchingWallLeft = knightPos.IsTouchingWallLeft();
        bool touchingWallRight = knightPos.IsTouchingWallRight();
        bool shadeCloakAvailable = knightPos.CanShadeCloakNow();

        bool isHitting = false;
        if (isAttacking && atkBox.HasValue)
        {
            foreach (Bounds hazard in boxes)
            {
                if (Overlaps2D(atkBox.Value, hazard))
                {
                    isHitting = true;
                    break;
                }
            }
        }

        bool isDamaged = false;
        foreach (Bounds hazard in boxes)
        {
            if (Overlaps2D(box, hazard))
            {
                isDamaged = true;
                break;
            }
        }

        lock (posLock)
        {
            knightPresent = true;
            knightBox = box;
            attacking = isAttacking;
            attackBox = atkBox;
            hazardBoxes = boxes;
            knightVelocity = velocity;
            knightFacingRight = facingRight;
            bossHealth = health;
            bossFacing = bossFacingRight.HasValue ? (bossFacingRight.Value ? 1 : 0) : -1;
            bossVelocity = bVelocity;
            knightHealth = kHealth;
            mana = manaAmount;

            isDashing = dashing;
            isJumping = jumping;
            isFalling = falling;
            isRecoiling = recoiling;
            isTouchingWallLeft = touchingWallLeft;
            isTouchingWallRight = touchingWallRight;
            canShadeCloakNow = shadeCloakAvailable;

            if (isHitting && !wasHitting)
                hitCount++;
            if (isDamaged && !wasDamaged && !wasInvulnerable)
                damageCount++;

            wasHitting = isHitting;
            wasDamaged = isDamaged;
            wasInvulnerable = invulnerable;
        }
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
        // via the OS backlog and then hung with no reply. Restarting main.py
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
    /// </summary>
    private string HandleInputCommand(string command)
    {
        string[] parts = command.Trim().Split(' ');
        string verb = parts[0].ToUpperInvariant();

        if (verb == "CLEAR" || verb == "KILL")
        {
            InputInjector.Enqueue(verb, null, 0f);
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
        public List<Bounds> HazardBoxes;
        public int Hits;
        public int Damages;
        public Vector2 KnightVelocity;
        public bool KnightFacingRight;
        public int BossHealth;
        public int BossFacing;
        public Vector2 BossVelocity;
        public int KnightHealth;
        public int Mana;
        public float GameSpeed;
        public bool Dashing;
        public bool Jumping;
        public bool Falling;
        public bool Recoiling;
        public bool TouchingWallLeft;
        public bool TouchingWallRight;
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
                HazardBoxes = hazardBoxes,
                Hits = hitCount,
                Damages = damageCount,
                KnightVelocity = knightVelocity,
                KnightFacingRight = knightFacingRight,
                BossHealth = bossHealth,
                BossFacing = bossFacing,
                BossVelocity = bossVelocity,
                KnightHealth = knightHealth,
                Mana = mana,
                GameSpeed = gameSpeed,
                Dashing = isDashing,
                Jumping = isJumping,
                Falling = isFalling,
                Recoiling = isRecoiling,
                TouchingWallLeft = isTouchingWallLeft,
                TouchingWallRight = isTouchingWallRight,
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
    ///                        boss_vel_x,boss_vel_y,knight_health,mana,speed
    ///   5:  flags            dashing,jumping,falling,recoiling,wall_left,
    ///                        wall_right,shade_cloak,invuln,paused
    ///   6+: one hazard box each, as section 0
    /// </summary>
    private string BuildStateLine()
    {
        StateSnapshot s = TakeSnapshot();

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
          .Append(s.GameSpeed);

        sb.Append(';')
          .Append(s.Dashing ? 1 : 0).Append(',')
          .Append(s.Jumping ? 1 : 0).Append(',')
          .Append(s.Falling ? 1 : 0).Append(',')
          .Append(s.Recoiling ? 1 : 0).Append(',')
          .Append(s.TouchingWallLeft ? 1 : 0).Append(',')
          .Append(s.TouchingWallRight ? 1 : 0).Append(',')
          .Append(s.ShadeCloakAvailable ? 1 : 0).Append(',')
          .Append(s.Invulnerable ? 1 : 0).Append(',')
          .Append(s.Paused ? 1 : 0);

        foreach (Bounds box in s.HazardBoxes)
        {
            sb.Append(';');
            AppendBox(sb, box);
        }

        return sb.ToString();
    }


    private void OnDestroy()
    {
        serverRunning = false;

        if (server != null)
            server.Stop();
    }


    private static bool Overlaps2D(Bounds a, Bounds b)
    {
        return a.min.x <= b.max.x && a.max.x >= b.min.x &&
               a.min.y <= b.max.y && a.max.y >= b.min.y;
    }
}
