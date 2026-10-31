using System;
using System.Collections;
using System.Net.Sockets;
using HarmonyLib;
using UnityEngine;

public partial class HollowRL
{
    private static HollowRL resetPlugin;
    internal static bool FastResetActive => resetPlugin != null && resetPlugin.fastResetEnabled;
    private readonly object resetGate = new object();
    private TcpClient resetOwner;
    private volatile bool fastResetEnabled;
    private bool resetPending;
    private bool resetDisablePending;
    private bool resetRunning;
    private bool resetDeathHeld;
    private int resetGeneration;
    private string resetState = "uncaptured";
    private string resetMessage = "Enter the configured boss fight once";
    private string resetScene;
    private string resetEntryGate;
    private int resetBossLevel;
    private int observedBossLevel = -1;
    private int resetHealth;
    private int resetBlueHealth;
    private int resetSoul;
    private int resetSoulReserve;
    private bool resetVisualRequested;
    private string resetVisual = "not sampled";

    // Capture the actual first fight, rather than hardcoding a scene, tier,
    // loadout, spawn gate or initial soul amount in Python.
    private void CaptureFightReset(HealthManager boss, int heroHealth, int soul)
    {
        if (boss == null || heroHealth <= 0 || boss.hp <= 0 || resetRunning
                || BossSceneController.Instance == null || GameManager.instance == null)
            return;
        lock (resetGate)
        {
            observedBossLevel = BossSceneController.Instance.BossLevel;
            if (resetScene != null) return;
            resetScene = boss.gameObject.scene.name;
            resetEntryGate = GameManager.instance.entryGateName;
            if (string.IsNullOrEmpty(resetEntryGate)) resetEntryGate = "door_dreamEnter";
            resetBossLevel = BossSceneController.Instance.BossLevel;
            resetHealth = heroHealth;
            resetBlueHealth = PlayerData.instance.healthBlue;
            resetSoul = soul;
            resetSoulReserve = PlayerData.instance.MPReserve;
            resetState = "idle";
            resetMessage = "captured";
        }
        Logger.LogInfo("HollowRL: reset target " + resetScene + " gate=" + resetEntryGate
            + " level=" + resetBossLevel + " hp=" + resetHealth + " soul=" + resetSoul);
    }

    private string HandleFightResetCommand(TcpClient client, string command)
    {
        string[] words = command.Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0 || !words[0].Equals("RESET_FIGHT", StringComparison.OrdinalIgnoreCase))
            return null;
        string operation = words.Length > 1 ? words[1].ToUpperInvariant() : "START";
        lock (resetGate)
        {
            if (operation == "STATUS")
                return "reset_v1;" + resetState + ";" + resetGeneration + ";"
                    + (resetScene ?? "") + ";" + resetMessage;
            if (operation == "VISUAL")
            {
                resetVisualRequested = true;
                return "visual_v1;" + resetVisual;
            }
            if (operation == "TARGET")
                return "target_v1;" + (resetScene ?? "") + ";" + resetBossLevel + ";" + observedBossLevel;
            if (DemonstrationActive) return "err human demonstration active";
            if (operation == "ENABLE")
            {
                if (resetOwner != null && resetOwner != client)
                    return "err another trainer owns fast reset";
                resetOwner = client;
                fastResetEnabled = true;
                return "ok reset_v1 enabled";
            }
            if (operation == "DISABLE")
            {
                if (resetOwner != null && resetOwner != client)
                    return "err another trainer owns fast reset";
                resetOwner = null;
                fastResetEnabled = false;
                resetDisablePending = true;
                return "ok reset_v1 disabled";
            }
            if (operation == "LEVEL")
            {
                int level;
                if (words.Length != 3 || !int.TryParse(words[2], out level) || level < 0 || level > 2)
                    return "err level must be 0 (Attuned), 1 (Ascended), or 2 (Radiant)";
                if (!fastResetEnabled || resetOwner != client) return "err fast reset is not enabled";
                if (resetScene == null) return "err reset target not captured";
                if (resetPending || resetRunning) return "err reset is busy";
                resetBossLevel = level;
                return "ok reset_v1 level " + level;
            }
            if (operation != "START") return "err unknown reset operation";
            if (!fastResetEnabled || resetOwner != client) return "err fast reset is not enabled";
            if (resetScene == null) return "err reset target not captured";
            if (resetPending || resetRunning)
                return "ok reset_v1 " + resetGeneration; // retry is idempotent while loading
            resetGeneration++;
            resetPending = true;
            resetState = "queued";
            resetMessage = "";
            return "ok reset_v1 " + resetGeneration;
        }
    }

    private void ReleaseFightResetOwner(TcpClient client)
    {
        lock (resetGate)
        {
            if (resetOwner != client) return;
            resetOwner = null;
            fastResetEnabled = false;
            resetDisablePending = true;
        }
    }

    private void FightResetTick()
    {
        bool begin = false;
        bool releaseDeath = false;
        bool sampleVisual = false;
        lock (resetGate)
        {
            if (resetDisablePending)
            {
                resetDisablePending = false;
                releaseDeath = resetDeathHeld && !resetRunning;
            }
            if (resetPending && !resetRunning)
            {
                resetPending = false;
                resetRunning = true;
                resetState = "loading";
                begin = true;
            }
            sampleVisual = resetVisualRequested;
            resetVisualRequested = false;
        }
        if (sampleVisual)
        {
            string visual = VisualState();
            lock (resetGate) resetVisual = visual;
        }
        if (releaseDeath && HeroController.instance != null)
        {
            resetDeathHeld = false;
            HeroController.instance.cState.dead = false;
            Time.timeScale = 1f;
            KillKnight(); // return to the game's normal death flow after disconnect
        }
        if (begin) StartCoroutine(GuardFightReset());
    }

    private IEnumerator GuardFightReset()
    {
        IEnumerator routine = ReloadFight();
        while (true)
        {
            object next = null;
            bool moved;
            try { moved = routine.MoveNext(); if (moved) next = routine.Current; }
            catch (Exception exc)
            {
                lock (resetGate)
                {
                    resetRunning = false;
                    resetState = "failed";
                    resetMessage = exc.Message.Replace(';', ',').Replace('\n', ' ').Replace('\r', ' ');
                    fastResetEnabled = false;
                    resetDisablePending = true;
                }
                Time.timeScale = 1f;
                Logger.LogError("HollowRL: RESET_FIGHT failed: " + exc);
                yield break;
            }
            if (!moved) yield break;
            yield return next;
        }
    }

    private IEnumerator ReloadFight()
    {
        InputInjector.ReleaseAll();
        Time.timeScale = 1f;
        GameManager manager = GameManager.instance;
        if (manager == null || HeroController.instance == null)
            throw new InvalidOperationException("Load a training save first");
        float deadline = Time.realtimeSinceStartup + 8f;
        while (manager.IsInSceneTransition)
        {
            if (Time.realtimeSinceStartup > deadline)
                throw new TimeoutException("An existing scene transition did not finish");
            yield return null;
        }
        HealthManager oldBoss = knightPos.GetBoss();
        int oldBossId = oldBoss != null ? oldBoss.GetInstanceID() : int.MinValue;
        HeroController hero = HeroController.instance;
        hero.StopAllCoroutines();
        // StopAllCoroutines does not execute Invulnerable's final cleanup.
        // A stale hazardInvulnRoutine makes StartInvulnerable skip setting
        // cState.invulnerable on subsequent hits, allowing contact damage
        // every physics tick. All three cached coroutine handles are now
        // stopped, so clear them before the next scene reuses this hero.
        foreach (var field in typeof(HeroController).GetFields(
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.NonPublic))
            if (field.FieldType == typeof(Coroutine)) field.SetValue(hero, null);
        Traverse.Create(hero).Field("invulnerableFreezeDuration").SetValue(0f);
        Traverse.Create(hero).Field("invulnerableDuration").SetValue(0f);
        Traverse.Create(hero).Field("parryInvulnTimer").SetValue(0f);
        var pulse = Traverse.Create(hero).Field("invPulse").GetValue<InvulnerablePulse>();
        if (pulse != null) pulse.stopInvulnerablePulse();
        hero.ResetState();
        Traverse.Create(hero).Method("CancelDamageRecoil").GetValue();
        Traverse.Create(hero).Method("ResetInput").GetValue();
        Traverse.Create(hero).Method("ResetMotion").GetValue();
        Traverse.Create(hero).Method("ResetAttacks").GetValue();
        hero.RegainControl();
        resetDeathHeld = false;
        RestoreResetResources();
        fightActive = false;
        knightDiedThisFight = false;
        bossDefeated = false;
        bossHealthTracked = false;
        knightHealthTracked = false;

        // This is the same one-shot setup callback used by BossChallengeUI.
        // Re-register it because BossSceneController consumes it in Awake.
        BossSceneController.SetupEvent = controller =>
        {
            controller.BossLevel = resetBossLevel;
            controller.DreamReturnEvent = "DREAM RETURN";
        };
        manager.BeginSceneTransition(new GameManager.SceneLoadInfo
        {
            SceneName = resetScene,
            EntryGateName = resetEntryGate,
            EntryDelay = 0f,
            WaitForSceneTransitionCameraFade = false,
            Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
        });
        Logger.LogInfo("HollowRL: RESET_FIGHT " + resetGeneration + " loading " + resetScene);

        deadline = Time.realtimeSinceStartup + 30f;
        bool restoredInNewScene = false;
        bool fadeRequested = false;
        float fadeRequestedAt = 0f;
        while (Time.realtimeSinceStartup < deadline)
        {
            HealthManager boss = knightPos.GetBoss();
            if (boss != null && boss.GetInstanceID() != oldBossId && boss.hp > 0
                    && HeroController.instance != null && !manager.IsInSceneTransition)
            {
                if (!restoredInNewScene)
                {
                    RestoreResetResources();
                    restoredInNewScene = true;
                }
                if (manager.HasFinishedEnteringScene && !fadeRequested)
                {
                    // The dream gate uses a custom fade. Direct scene loads
                    // must explicitly pair BeginSceneTransition's FADE OUT
                    // with the native FADE SCENE IN event.
                    Logger.LogInfo("HollowRL: reset visuals before fade " + VisualState());
                    manager.FadeSceneIn();
                    fadeRequested = true;
                    fadeRequestedAt = Time.realtimeSinceStartup;
                }
                string state = ReadControlFsmState(boss);
                if (manager.HasFinishedEnteringScene && !string.IsNullOrEmpty(state)
                        && !ResetIntroStates.Contains(state)
                        && knightPos.GetBossCollider(boss) != null
                        && PlayerData.instance.health > 0 && fadeRequested
                        && Time.realtimeSinceStartup - fadeRequestedAt >= 0.25f
                        && SceneVisualReady())
                {
                    InputInjector.ReleaseAll();
                    if (fastResetEnabled) Time.timeScale = 0f;
                    yield return null; // publish the new fight before reporting ready
                    lock (resetGate)
                    {
                        resetRunning = false;
                        resetState = "ready";
                        resetMessage = "combat ready";
                    }
                    Logger.LogInfo("HollowRL: RESET_FIGHT " + resetGeneration + " ready " + VisualState());
                    yield break;
                }
            }
            yield return null;
        }
        throw new TimeoutException("New boss scene did not become visually/combat ready: " + VisualState());
    }

    private static bool SceneVisualReady()
    {
        GameCameras cameras = GameCameras.instance;
        if (cameras == null || cameras.mainCamera == null
                || !cameras.mainCamera.isActiveAndEnabled || cameras.cameraFadeFSM == null)
            return false;
        PlayMakerFSM fade = cameras.cameraFadeFSM;
        Component sprite = fade.GetComponent("tk2dSprite");
        if (sprite != null && Traverse.Create(sprite).Property("color").GetValue<Color>().a > 0.02f) return false;
        SpriteRenderer renderer = fade.GetComponent<SpriteRenderer>();
        if (renderer != null && renderer.enabled && renderer.color.a > 0.02f) return false;
        return string.Equals(fade.ActiveStateName, "Normal", StringComparison.OrdinalIgnoreCase);
    }

    private static string VisualState()
    {
        GameCameras cameras = GameCameras.instance;
        if (cameras == null) return "no cameras";
        PlayMakerFSM fade = cameras.cameraFadeFSM;
        string result = "scene=" + (GameManager.instance != null ? GameManager.instance.sceneName : "none")
            + " main=" + (cameras.mainCamera != null && cameras.mainCamera.isActiveAndEnabled)
            + " fade=" + (fade != null ? fade.ActiveStateName : "none");
        if (fade != null)
        {
            Component sprite = fade.GetComponent("tk2dSprite");
            if (sprite != null) result += " alpha=" + Traverse.Create(sprite).Property("color").GetValue<Color>().a;
            foreach (var value in fade.FsmVariables.FloatVariables)
                result += " " + value.Name + "=" + value.Value;
        }
        if (cameras.mainCamera != null) result += " cameraPos=" + cameras.mainCamera.transform.position;
        if (HeroController.instance != null) result += " heroPos=" + HeroController.instance.transform.position;
        return result.Replace(';', ',').Replace('\n', ' ').Replace('\r', ' ');
    }

    private void RestoreResetResources()
    {
        PlayerData data = PlayerData.instance;
        data.health = resetHealth;
        data.healthBlue = resetBlueHealth;
        data.MPCharge = resetSoul;
        data.MPReserve = resetSoulReserve;
        data.isInvincible = false;
        data.disablePause = false;
        HeroBox.inactive = false;
    }

    private static readonly System.Collections.Generic.HashSet<string> ResetIntroStates =
        new System.Collections.Generic.HashSet<string> {
            "Init", "Emerge Dust", "Fall", "Intro Land", "Roar", "Roar End", "Roar Recover"
        };

    private bool HoldTrainingDeath(HeroController hero)
    {
        if (!fastResetEnabled || resetScene == null || resetRunning || DemonstrationActive
                || GameManager.instance == null || GameManager.instance.sceneName != resetScene)
            return false;
        hero.RelinquishControl();
        hero.cState.dead = true;
        InputInjector.ReleaseAll();
        PlayerData.instance.health = 0;
        resetDeathHeld = true;
        Time.timeScale = 0f;
        return true;
    }

    private static IEnumerator EmptyDeath() { yield break; }

    [HarmonyPatch(typeof(HeroController), "Die")]
    private static class TrainingDeathPatch
    {
        private static bool Prefix(HeroController __instance, ref IEnumerator __result)
        {
            if (resetPlugin == null || !resetPlugin.HoldTrainingDeath(__instance)) return true;
            __result = EmptyDeath();
            return false;
        }
    }
}
