from numpy import argmax
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import calculate_damage, Move, Category, Type


class SwitchExploitPolicy(BattlePolicy):
    """
    An agent that uses a Turn-to-KO (TTKO) simulation to decide between
    staying in or switching to a benched Pokémon.
    """

    def get_best_move_dmg(self, attacker, defender, state) -> float:
        """Calculates the maximum possible damage the attacker can deal this turn."""
        outcomes = []
        for bm in attacker.battling_moves:
            if bm.pp > 0 and not bm.disabled:
                # params, side_index, move, state, attacker, defender
                dmg = calculate_damage(self.params, 0, bm.constants, state, attacker, defender)
                outcomes.append(dmg)
        return max(outcomes) if outcomes else 0

    def estimate_ttko(self, attacker, defender, state, is_switch: bool) -> float:
        """
        Estimates the number of turns to KO the opponent.
        Returns 99.0 if the attacker is KO'd before securing the win.
        """
        dmg_dealt = self.get_best_move_dmg(attacker, defender, state)
        if dmg_dealt <= 0:
            return 99

        # FIXED HEURISTIC: Check if opponent is Physical or Special
        # This prevents the "99" hallucination if the bench has lopsided defenses
        is_special = defender.constants.stats[3] > defender.constants.stats[1]
        category = Category.SPECIAL if is_special else Category.PHYSICAL

        opp_dmg_options = []
        for t in defender.types:
            if t is not None:
                # We use the actual defender's stats to simulate their best STAB move
                opp_move = Move(pkm_type=t, base_power=90, accuracy=1.0,
                                max_pp=10, category=category)
                # CRITICAL: We calculate damage against the specific 'attacker' (which might be the benched pkm)
                opp_dmg_options.append(calculate_damage(self.params, 1, opp_move, state, defender, attacker))

        dmg_taken = max(opp_dmg_options) if opp_dmg_options else 10

        curr_hp = attacker.hp
        opp_hp = defender.hp
        turns = 0

        # Turn 0: The Switch turn
        if is_switch:
            curr_hp -= dmg_taken
            turns += 1
            if curr_hp <= 0:
                return 99

        my_spd = attacker.constants.stats[5]
        opp_spd = defender.constants.stats[5]

        # Simulate battle race
        while curr_hp > 0 and opp_hp > 0 and turns < 20:
            if my_spd >= opp_spd:
                # We hit first
                opp_hp -= dmg_dealt
                if opp_hp <= 0:
                    return turns + 1  # Added 1 for the move turn
                curr_hp -= dmg_taken
            else:
                # They hit first
                curr_hp -= dmg_taken
                if curr_hp <= 0:
                    return 99
                opp_hp -= dmg_dealt

            turns += 1

        return turns if opp_hp <= 0 else 99

    def decision(self, state, opp_view=None) -> list:
        # Navigate the VGC2 state structure
        my_side = state.sides[0]
        opp_side = state.sides[1]

        active = my_side.team.active[0]
        # print("my active", active)
        opp_active = opp_side.team.active[0]
        bench = my_side.team.reserve

        # 1. Evaluate Staying In
        stay_ttko = self.estimate_ttko(active, opp_active, state, is_switch=False)

        # 2. Evaluate Switching
        best_switch_ttko = 99
        best_switch_idx = -1

        for i, benched_pkm in enumerate(bench):
            if benched_pkm.hp > 0:
                s_ttko = self.estimate_ttko(benched_pkm, opp_active, state, is_switch=True)
                if s_ttko < best_switch_ttko:
                    best_switch_ttko = s_ttko
                    best_switch_idx = i

        # Debugging prints to monitor the 'Race'
        # print(f"{stay_ttko} | {best_switch_ttko}")

        # 3. Decision Logic: Switch ONLY if it strictly improves the clock
        if best_switch_idx != -1 and best_switch_ttko < stay_ttko:
            # Action -1 is switch, target is the reserve index
            # print("SWITCH")
            return [(-1, best_switch_idx)]

        # 4. Fallback: Greedy Attack selection
        move_outcomes = [
            calculate_damage(self.params, 0, bm.constants, state, active, opp_active)
            if bm.pp > 0 and not bm.disabled else 0
            for bm in active.battling_moves
        ]

        # If no damage can be done, default to first move or struggle
        best_move = int(argmax(move_outcomes)) if any(move_outcomes) else 0
        return [(best_move, 0)]