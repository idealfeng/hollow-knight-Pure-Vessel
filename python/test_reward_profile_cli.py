import sys
import unittest
from unittest.mock import patch

from main_multi import parse_args


class RewardProfileCliTests(unittest.TestCase):
    def test_wave_clear_trial_arguments_remain_explicit_and_compatible(self):
        with patch.object(sys, "argv", ["main_multi.py", "--reward-profile", "radiant_wave_clear_v1",
                "--boss-level", "2", "--episode-time-limit", "120", "--observation-time-limit", "60",
                "--eval-objective", "nohit", "--ent-coef", "0.005", "--lr", "0.0001"]):
            args = parse_args()
        self.assertEqual(args.reward_profile, "radiant_wave_clear_v1")
        self.assertEqual(args.boss_level, 2)
        self.assertEqual((args.episode_time_limit, args.observation_time_limit), (120, 60))
        self.assertEqual((args.ent_coef, args.lr), (0.005, 0.0001))

    def test_existing_reward_profiles_and_inherited_default_still_parse(self):
        for profile in (None, "victory_v1", "nohit_v1", "radiant_spacing_v1"):
            with self.subTest(profile=profile), patch.object(sys, "argv",
                    ["main_multi.py"] + (["--reward-profile", profile] if profile else [])):
                self.assertEqual(parse_args().reward_profile, profile)


if __name__ == "__main__":
    unittest.main()
