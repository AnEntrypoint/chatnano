"""Assign a public-domain reference voice to each hero by archetype and emit a manifest."""
import json, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hero_lines import LINES, EVENTS

DEEP = ["axe","tidehunter","sand_king","earthshaker","lich","enigma","warlock","undying",
        "centaur","pudge","slardar","bristleback","doom_bringer","spirit_breaker","lycan",
        "ursa","ogre_magi","tiny","kunkka","beastmaster","abaddon","skeleton_king",
        "elder_titan","magnataur","primal_beast","troll_warlord","huskar","alchemist",
        "night_stalker","brewmaster","tusk","meepo"]
MID  = ["juggernaut","antimage","bloodseeker","riki","sniper","viper","templar_assassin",
        "slark","ember_spirit","storm_spirit","invoker","phantom_lancer","morphling",
        "shadow_shaman","bane","rattletrap","tinker","death_prophet","necrolyte","venomancer"]
HIGH = ["crystal_maiden","drow_ranger","mirana","lina","queenofpain","medusa","naga_siren",
        "legion_commander","winter_wyvern","dark_willow","hoodwink","marci","muerta",
        "windrunner","enchantress","broodmother","vengefulspirit","phantom_assassin",
        "spectre","witch_doctor","nyx_assassin"]

def main():
    heroes = [h.strip() for h in open('/tmp/heroes_wired.txt') if h.strip()]
    assign, deep, mid, high = {}, itertools.cycle(["dg","mlo","wt","ik"]), \
        itertools.cycle(["cjph","lcw","el"]), itertools.cycle(["cc","ks","asm","bk","ss"])
    for h in heroes:
        if h in DEEP:   assign[h] = next(deep)
        elif h in MID:  assign[h] = next(mid)
        elif h in HIGH: assign[h] = next(high)
        else:           assign[h] = next(mid)
    chars = [{"name": h, "ref": f"voices/refs/pd_{assign[h]}.wav", "lines": LINES[h]}
             for h in heroes]
    json.dump({"out_dir": "voices/out", "events": EVENTS, "characters": chars},
              open("voices/manifest.json", "w"), indent=1)
    json.dump({"events": EVENTS,
               "voices": {h: {"voice_ref": f"pd_{assign[h]}",
                              **{e: f"voices/out/{h}/{i:03d}.wav" for i, e in enumerate(EVENTS)}}
                          for h in heroes}},
              open("voices/voices.manifest.json", "w"), indent=1)
    from collections import Counter
    print("heroes:", len(heroes), "| unique line count:",
          len({l for c in chars for l in c['lines']}))
    print("ref usage:", dict(Counter(assign.values())))

main()
