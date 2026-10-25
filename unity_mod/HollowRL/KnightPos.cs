using System.Collections.Generic;
using HarmonyLib;
using UnityEngine;


namespace KnightPos
{
    public struct HazardSnapshot
    {
        public int InstanceId;
        public string ObjectName;
        public string RootName;
        public string HierarchyPath;
        public Bounds Box;
        public Vector2 Velocity;
        public bool BossAttached;
    }

    public class DiagnosticsOverlay : MonoBehaviour
    {
        private const string PureVesselObjectName = "HK Prime";
        private const string PureVesselSceneName = "GG_Hollow_Knight";
        private const float BossSearchInterval = 0.25f;

        private HealthManager cachedBoss;
        private float nextBossSearchAt;

        private struct HazardMotion
        {
            public Vector2 Center;
            public float GameTime;
        }

        private readonly Dictionary<int, HazardMotion> hazardMotion =
            new Dictionary<int, HazardMotion>();

        public Bounds GetKnightBox()
        {
            return HeroController.instance.GetComponent<Collider2D>().bounds;
        }

        public bool IsAttacking()
        {
            return HeroController.instance.cState.attacking;
        }

        public bool IsInvulnerable()
        {
            return HeroController.instance.cState.invulnerable;
        }

        public bool IsDashing()
        {
            return HeroController.instance.cState.dashing;
        }

        public bool IsJumping()
        {
            return HeroController.instance.cState.jumping;
        }

        public bool IsFalling()
        {
            return HeroController.instance.cState.falling;
        }

        public bool IsOnGround()
        {
            // Unlike the old Python comparison against a Hornet-arena Y
            // constant, this is the game's authoritative grounded state and
            // remains valid in every scene and on uneven platforms.
            return HeroController.instance.cState.onGround;
        }

        public bool IsRecoiling()
        {
            return HeroController.instance.cState.recoiling;
        }

        // touchingWallL/R are private fields on HeroController (not on cState),
        // so they're read via reflection rather than a direct field access.
        public bool IsTouchingWallLeft()
        {
            return Traverse.Create(HeroController.instance).Field("touchingWallL").GetValue<bool>();
        }

        public bool IsTouchingWallRight()
        {
            return Traverse.Create(HeroController.instance).Field("touchingWallR").GetValue<bool>();
        }

        public Vector2 GetWallDistances()
        {
            // Pure Vessel's arena is flat, but absolute world X still makes a
            // poor policy input. Raycast only against terrain so enemies and
            // temporary attack objects cannot masquerade as a wall.
            int terrainLayer = LayerMask.NameToLayer("Terrain");
            if (terrainLayer < 0)
                return new Vector2(100f, 100f);

            int terrainMask = 1 << terrainLayer;
            Vector2 origin = GetKnightBox().center;
            RaycastHit2D left = Physics2D.Raycast(origin, Vector2.left, 100f, terrainMask);
            RaycastHit2D right = Physics2D.Raycast(origin, Vector2.right, 100f, terrainMask);
            return new Vector2(
                left.collider != null ? left.distance : 100f,
                right.collider != null ? right.distance : 100f);
        }

        // Shade Cloak is tracked internally as "hasShadowDash" (not
        // "hasSuperDash" = Crystal Heart, and not "hasDashSlash" = the Dash
        // Slash nail art - both dead ends from earlier). It has its own
        // separate cooldown timer, "shadowDashTimer" - HeroDash() only procs
        // the shadow-dash effect when "hasShadowDash && shadowDashTimer <= 0f",
        // so that's exactly "can Shade Cloak trigger on the next dash".
        // shadowDashTimer is private, so read via reflection.
        public bool CanShadeCloakNow()
        {
            bool hasShadowDash = PlayerData.instance.GetBool("hasShadowDash");
            float shadowDashTimer = Traverse.Create(HeroController.instance).Field("shadowDashTimer").GetValue<float>();
            return hasShadowDash && shadowDashTimer <= 0f;
        }

        public Vector2 GetKnightVelocity()
        {
            Rigidbody2D rb = HeroController.instance.GetComponent<Rigidbody2D>();
            return rb != null ? rb.velocity : Vector2.zero;
        }

        public bool IsKnightFacingRight()
        {
            return HeroController.instance.cState.facingRight;
        }

        public int GetKnightHealth()
        {
            return PlayerData.instance.health;
        }

        // MPCharge is the current soul/mana amount (0-99 per vessel, spends in
        // units of 33 on a spell cast or a heal). MPReserve is overflow soul
        // beyond one vessel (extra Vessel charms) - MPCharge alone is what
        // actually gates "can I cast/heal right now".
        public int GetMana()
        {
            return PlayerData.instance.MPCharge;
        }

        public HealthManager GetBoss()
        {
            // Live diagnostics confirmed Pure Vessel is the object named
            // "HK Prime" in scene "GG_Hollow_Knight". Keep the reference
            // while it is alive, then reacquire after a death/retry creates a
            // new Unity instance. Outside the arena, limit scene-wide scans.
            if (IsPureVessel(cachedBoss))
                return cachedBoss;

            cachedBoss = null;
            if (Time.unscaledTime < nextBossSearchAt)
                return null;

            nextBossSearchAt = Time.unscaledTime + BossSearchInterval;
            foreach (HealthManager candidate in FindObjectsOfType<HealthManager>())
            {
                if (IsPureVessel(candidate))
                {
                    cachedBoss = candidate;
                    break;
                }
            }

            return cachedBoss;
        }

        private static bool IsPureVessel(HealthManager candidate)
        {
            return candidate != null
                && candidate.gameObject.activeInHierarchy
                && candidate.gameObject.name == PureVesselObjectName
                && candidate.gameObject.scene.name == PureVesselSceneName;
        }

        // These three used to each call GetBoss() independently, so a single
        // Update() frame did three separate FindObjectOfType<HealthManager>()
        // scene scans for the same object - a call Unity's own docs say not
        // to make even once per frame. Callers now fetch the boss once (via
        // GetBoss()) and pass it in.
        public int? GetBossHealth(HealthManager boss)
        {
            // null means "no boss object to read right now" - NOT 0 hp. The
            // caller must distinguish "missing" from "defeated" itself.
            return boss != null ? boss.hp : (int?)null;
        }

        public bool? IsBossFacingRight(HealthManager boss)
        {
            if (boss == null)
                return null;

            return boss.transform.localScale.x >= 0;
        }

        public Vector2 GetBossVelocity(HealthManager boss)
        {
            if (boss == null)
                return Vector2.zero;

            Rigidbody2D rb = boss.GetComponent<Rigidbody2D>();
            return rb != null ? rb.velocity : Vector2.zero;
        }

        public Vector2 GetBossPosition(HealthManager boss)
        {
            return boss != null ? (Vector2)boss.transform.position : Vector2.zero;
        }

        public Bounds? GetBossBox(HealthManager boss)
        {
            if (boss == null)
                return null;

            Collider2D collider = boss.GetComponent<Collider2D>();
            if (collider == null)
                collider = boss.GetComponentInChildren<Collider2D>();

            return collider != null && collider.enabled ? collider.bounds : (Bounds?)null;
        }

        // HeroController keeps 5 separate NailSlash instances as fields
        // (normalSlash, alternateSlash, upSlash, downSlash, wallSlash) and picks
        // one per swing into the private "slashComponent" field. GetComponentInChildren
        // <NailSlash>() ignores that choice and always returns whichever NailSlash
        // happens to be first in the hierarchy - so up/down slashes were reading
        // the wrong (disabled) collider and coming back null. Read the actual
        // active one instead.
        public Bounds? GetAttackBox()
        {
            if (!IsAttacking())
                return null;

            NailSlash slash = Traverse.Create(HeroController.instance).Field("slashComponent").GetValue<NailSlash>();
            if (slash == null)
                return null;

            Collider2D col = slash.GetComponent<Collider2D>();
            if (col == null || !col.enabled)
                return null;

            return col.bounds;
        }

        public List<HazardSnapshot> GetHazards(HealthManager boss)
        {
            DamageHero[] currentHazards = FindObjectsOfType<DamageHero>();
            var snapshots = new List<HazardSnapshot>();
            var liveIds = new HashSet<int>();
            float now = Time.time;

            foreach (DamageHero hazard in currentHazards)
            {
                if (hazard == null || !hazard.gameObject.activeInHierarchy)
                    continue;

                Collider2D col = hazard.GetComponent<Collider2D>();
                if (col == null || !col.enabled)
                    continue;

                int id = hazard.GetInstanceID();
                liveIds.Add(id);
                Bounds box = col.bounds;
                Vector2 center = box.center;
                Vector2 velocity = Vector2.zero;

                Rigidbody2D rb = hazard.GetComponent<Rigidbody2D>();
                if (rb != null)
                {
                    velocity = rb.velocity;
                }
                else
                {
                    HazardMotion previous;
                    if (hazardMotion.TryGetValue(id, out previous))
                    {
                        float dt = now - previous.GameTime;
                        if (dt > 0.0001f)
                            velocity = (center - previous.Center) / dt;
                    }
                }

                hazardMotion[id] = new HazardMotion { Center = center, GameTime = now };

                Transform root = hazard.transform.root;
                snapshots.Add(new HazardSnapshot
                {
                    InstanceId = id,
                    ObjectName = hazard.gameObject.name,
                    RootName = root != null ? root.name : "",
                    HierarchyPath = GetHierarchyPath(hazard.transform),
                    Box = box,
                    Velocity = velocity,
                    BossAttached = boss != null
                        && (hazard.transform == boss.transform
                            || hazard.transform.IsChildOf(boss.transform)),
                });
            }

            var staleIds = new List<int>();
            foreach (int id in hazardMotion.Keys)
            {
                if (!liveIds.Contains(id))
                    staleIds.Add(id);
            }
            foreach (int id in staleIds)
                hazardMotion.Remove(id);

            return snapshots;
        }

        private static string GetHierarchyPath(Transform node)
        {
            var names = new List<string>();
            while (node != null)
            {
                names.Add(node.name);
                node = node.parent;
            }
            names.Reverse();
            return string.Join("/", names.ToArray());
        }

        private bool ShowColliders = true;
        private Material _lineMaterial;
        private GUIStyle _labelStyle;
        private bool LockKnightHealth = false;

        // F8 toggles a debug health lock so Pure Vessel's moves can be watched
        // without dying to the first few attacks. Debug only - training assumes
        // the knight can actually die.
        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.F8))
                LockKnightHealth = !LockKnightHealth;

            if (LockKnightHealth
                && HeroController.instance != null
                && PlayerData.instance != null
                && PlayerData.instance.health < 9)
            {
                PlayerData.instance.health = 9;
            }
        }


        // Debug overlay: draw every relevant collider as a wireframe box in the
        // game view so move handling can be eyeballed directly. F9 toggles it.
        // Green=knight, red=boss body, cyan=nail hitbox, yellow=detached hazard,
        // magenta=hazard attached to the boss.
        private void OnGUI()
        {
            if (Event.current.type == EventType.KeyDown
                && Event.current.keyCode == KeyCode.F9)
            {
                ShowColliders = !ShowColliders;
            }

            if (!ShowColliders)
                return;

            // The knight only exists once a save is loaded; drawing before that
            // throws on every frame and corrupts the GL/matrix state (black screen).
            if (HeroController.instance == null)
                return;

            Camera cam = Camera.main;
            if (cam == null)
                return;

            if (_lineMaterial == null)
                _lineMaterial = new Material(Shader.Find("Hidden/Internal-Colored"));

            if (_labelStyle == null)
            {
                _labelStyle = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 13,
                };
                _labelStyle.normal.textColor = Color.white;
            }

            HealthManager boss = GetBoss();

            if (Event.current.type == EventType.Repaint)
            {
                _lineMaterial.SetPass(0);
                GL.PushMatrix();
                GL.LoadPixelMatrix(0, cam.pixelWidth, cam.pixelHeight, 0);
                try
                {
                    GL.Begin(GL.LINES);

                    DrawBox(cam, GetKnightBox(), Color.green);

                    if (boss != null)
                    {
                        Bounds? bossBox = GetBossBox(boss);
                        if (bossBox.HasValue)
                            DrawBox(cam, bossBox.Value, Color.red);
                    }

                    if (IsAttacking())
                    {
                        Bounds? atk = GetAttackBox();
                        if (atk.HasValue)
                            DrawBox(cam, atk.Value, Color.cyan);
                    }

                    foreach (HazardSnapshot hazard in GetHazards(boss))
                    {
                        Color color = hazard.BossAttached ? Color.magenta : Color.yellow;
                        DrawBox(cam, hazard.Box, color);
                    }
                }
                finally
                {
                    GL.End();
                    GL.PopMatrix();
                }
            }

            foreach (HazardSnapshot hazard in GetHazards(boss))
            {
                DrawLabel(cam, hazard.Box.center, hazard.ObjectName, Color.yellow);
            }
        }

        private void DrawBox(Camera cam, Bounds box, Color color)
        {
            Vector2 p0 = ToScreen(cam, new Vector3(box.min.x, box.min.y, 0));
            Vector2 p1 = ToScreen(cam, new Vector3(box.max.x, box.min.y, 0));
            Vector2 p2 = ToScreen(cam, new Vector3(box.max.x, box.max.y, 0));
            Vector2 p3 = ToScreen(cam, new Vector3(box.min.x, box.max.y, 0));

            DrawLine(p0, p1, color);
            DrawLine(p1, p2, color);
            DrawLine(p2, p3, color);
            DrawLine(p3, p0, color);
        }

        private void DrawLine(Vector2 a, Vector2 b, Color color)
        {
            GL.Color(color);
            GL.Vertex3(a.x, a.y, 0);
            GL.Vertex3(b.x, b.y, 0);
        }

        private static Vector2 ToScreen(Camera cam, Vector3 world)
        {
            Vector3 sp = cam.WorldToScreenPoint(world);
            return new Vector2(sp.x, cam.pixelHeight - sp.y);
        }

        private void DrawLabel(Camera cam, Vector3 worldPos, string text, Color color)
        {
            GUIContent content = new GUIContent(text);
            Vector2 size = _labelStyle.CalcSize(content);
            Vector2 gui = ToScreen(cam, worldPos);
            Rect rect = new Rect(gui.x - size.x / 2f, gui.y - size.y - 2f,
                                 size.x + 4f, size.y);
            _labelStyle.normal.textColor = color;
            GUI.Label(rect, content, _labelStyle);
        }

    }
}
