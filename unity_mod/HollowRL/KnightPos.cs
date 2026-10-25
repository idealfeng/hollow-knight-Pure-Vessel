using System.Collections.Generic;
using HarmonyLib;
using UnityEngine;


namespace KnightPos
{
    public class DiagnosticsOverlay : MonoBehaviour
    {
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
            return FindObjectOfType<HealthManager>();
        }

        // These three used to each call GetBoss() independently, so a single
        // Update() frame did three separate FindObjectOfType<HealthManager>()
        // scene scans for the same object - a call Unity's own docs say not
        // to make even once per frame. Callers now fetch the boss once (via
        // GetBoss()) and pass it in.
        public int GetBossHealth(HealthManager boss)
        {
            return boss != null ? boss.hp : 0;
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

        private readonly List<DamageHero> trackedHazards = new List<DamageHero>();

        public List<Bounds> GetHazardBoxes()
        {
            DamageHero[] currentHazards = FindObjectsOfType<DamageHero>();

            // drop hazards that no longer exist, keep the rest in their original order
            trackedHazards.RemoveAll(h => h == null || System.Array.IndexOf(currentHazards, h) < 0);

            // newly seen hazards go to the back of the queue
            foreach (DamageHero hazard in currentHazards)
            {
                if (!trackedHazards.Contains(hazard))
                    trackedHazards.Add(hazard);
            }

            var boxes = new List<Bounds>();

            foreach (DamageHero hazard in trackedHazards)
            {
                Collider2D col = hazard.GetComponent<Collider2D>();
                if (col != null)
                    boxes.Add(col.bounds);
            }

            return boxes;
        }
    }
}