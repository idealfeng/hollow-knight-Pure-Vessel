from dataclasses import asdict
import unittest

from agent.reward import RewardTracker, calculate_reward, get_reward_config
from game_state import Hazard
from .test_reward import make_state


def state(fsm, x=None, identifier=1, **kwargs):
    hazards = [] if x is None else [Hazard(identifier, (x, 0, x+2.5, 15), (-14.5, 0), False)]
    snapshot = make_state(fsm=fsm, boss_hp=1300, knight_hp=9,
                          knight_box=(0, 1, 1, 3), boss_box=(7, 0, 10, 4), **kwargs)
    snapshot["hazards"] = hazards
    return snapshot


class WaveClearRewardTests(unittest.TestCase):
    def complete(self, tracker, identifier=1, end_flags=None):
        snapshots = [state("Slam Antic"), state("Slamming", 6, identifier),
                     state("Waves", 2, identifier), state("Slam End", -5, identifier),
                     state("Idle", -7, identifier)]
        parts = []
        for index, (previous, current) in enumerate(zip(snapshots,snapshots[1:])):
            tracker.transition(previous,current,done=False,
                               **(end_flags if end_flags and index==3 else {}))
            parts.append(dict(tracker.last_breakdown))
        return parts

    def test_profile_changes_only_wave_shaping(self):
        old = asdict(get_reward_config("radiant_spacing_v1"))
        new = asdict(get_reward_config("radiant_wave_clear_v1"))
        difference = {k for k in old if old[k] != new[k]}
        self.assertEqual(difference, {"profile","shockwave_spacing_bonus","shockwave_clear_bonus"})
        self.assertEqual(new["shockwave_clear_bonus"],1)
        self.assertEqual(new["shockwave_clear_max_rewards"],2)
        self.assertEqual((new["damage_per_mask"],new["death"],new["boss_defeated"]),(3,-20,100))

    def test_real_pass_awards_once_after_clearance(self):
        tracker=RewardTracker(get_reward_config("radiant_wave_clear_v1"))
        parts=self.complete(tracker)
        self.assertTrue(all("shockwave clear" not in p for p in parts[:-1]))
        self.assertEqual(parts[-1]["shockwave clear"],1)
        self.assertEqual(tracker.wave_clear_rewards,1)
        self.assertTrue(all("shockwave spacing" not in p for p in parts))

    def test_episode_cap_and_reset(self):
        tracker=RewardTracker(get_reward_config("radiant_wave_clear_v1"))
        awards=[]
        for identifier in (1,2,3):
            awards.extend(p.get("shockwave clear",0) for p in self.complete(tracker,identifier))
        self.assertEqual(sum(awards),2)
        self.assertEqual(tracker.wave_clear_rewards,2)
        tracker.reset()
        self.assertEqual(self.complete(tracker,4)[-1]["shockwave clear"],1)

    def test_existing_profiles_never_create_new_detector_or_bonus(self):
        for profile in ("victory_v1","nohit_v1","radiant_spacing_v1"):
            with self.subTest(profile=profile):
                tracker=RewardTracker(get_reward_config(profile))
                self.assertIsNone(tracker._wave_clearance)
                self.assertTrue(all("shockwave clear" not in p for p in self.complete(tracker)))

    def test_no_bonus_for_terminal_damage_timeout_or_truncated_transition(self):
        config=get_reward_config("radiant_wave_clear_v1")
        previous=state("Waves"); current=state("Idle")
        for changes,flags in [({}, {"done":True}), ({},{"timed_out":True}),
                              ({},{"truncated":True}),
                              ({"knight_health":0,"damage_count":9},{}),
                              ({"knight_health":8,"damage_count":1},{})]:
            with self.subTest(changes=changes,flags=flags):
                kwargs={"done":False,**flags}
                _,parts=calculate_reward(previous,{**current,**changes},reward_config=config,
                                         successful_wave_clear=True,return_breakdown=True,**kwargs)
                self.assertNotIn("shockwave clear",dict(parts))
        tracker=RewardTracker(config)
        self.assertTrue(all("shockwave clear" not in p for p in self.complete(tracker,end_flags={"truncated":True})))


if __name__ == "__main__":
    unittest.main()
