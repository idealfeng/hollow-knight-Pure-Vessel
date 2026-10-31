import unittest
from policy_selection import is_flawless, nohit_candidate_improved, nohit_selection_score


def result(wins=0.975, nohit=0.10, damage=2.75):
    return {"win_rate": wins, "nohit_rate": nohit, "mean_damage": damage, "mean_min_boss_hp": 1.85}


class NoHitSelectionTests(unittest.TestCase):
    def test_counter_reset_or_recovery_is_not_a_flawless_win(self):
        self.assertFalse(is_flawless({"won": True, "damage": 0, "episode_damage": 2}))
        self.assertFalse(is_flawless({"won": True, "damage": 0, "damage_tracking_valid": False}))
        self.assertTrue(is_flawless({"won": True, "damage": 0}))

    def test_nohit_model_beats_perfect_win_model_inside_gate(self):
        self.assertGreater(nohit_selection_score(result()),
                           nohit_selection_score(result(wins=1.0, nohit=0, damage=3.3)))

    def test_cannot_win_selection_by_giving_up_fights(self):
        self.assertLess(nohit_selection_score(result(wins=0.5, nohit=0.5, damage=1)),
                        nohit_selection_score(result()))
        self.assertFalse(nohit_candidate_improved(result(wins=0.5, nohit=0.5, damage=1), result()))

    def test_more_nohits_must_not_trade_for_more_damage(self):
        self.assertFalse(nohit_candidate_improved(result(nohit=0.20, damage=3.0), result()))
        self.assertTrue(nohit_candidate_improved(result(nohit=0.20, damage=2.5), result()))

    def test_lower_damage_can_be_intermediate_progress(self):
        self.assertTrue(nohit_candidate_improved(result(damage=2.0), result()))
        self.assertFalse(nohit_candidate_improved(result(damage=2.6), result()))


if __name__ == "__main__":
    unittest.main()
