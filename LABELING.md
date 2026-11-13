# Labeling guide

One crop = one complete symbol. A class means symbol AND color (color carries urgency).

| Folder | What to look for | Color | Not this (put in `none`) |
|---|---|---|---|
| `high_beam` | "D"-shaped headlamp with straight horizontal lines | Blue | Amber or green "A" auto-high-beam symbol |
| `headlights` | Same "D" lamp, lines slanting downward (low beam) | Green | Position lights (two small lamps); amber "A" auto-headlight |
| `turn_left` | Arrow pointing left (solid, or outline while fading) | Green | Hazard triangle |
| `turn_right` | Arrow pointing right | Green | Hazard triangle |
| `abs` | Circle with the letters "ABS" and arcs at the sides | Amber | Traction or skid-control car icons; "OFF" labels |
| `brake_system` | Circled "!" with arcs at the sides; often the word BRAKE | Red | Amber circled "!" |
| `parking_brake` | Circled "P" with arcs at the sides; often the word PARK | Red | Green "HOLD" or auto-hold |
| `battery` | Rectangle with + and - terminals on top | Red | |
| `oil_pressure` | Old-fashioned oil can with a drip | Red | Amber oil can with wavy line (oil level) |
| `check_engine` | Engine-block outline | Amber | |
| `tire_pressure` | Flat-bottomed horseshoe (tire cross-section) with "!" inside | Amber | Car with skid marks |
| `airbag` | Seated person with a large circle in front | Red | Person with a diagonal strap (seat belt) |
| `none` | Any complete lit symbol or text NOT in the table | any | See list below |

`none` examples: seat belt, door or trunk open, key or fob, steering wheel with "!", wrench, AWD, AT, HOLD, OFF, ECO,
traction or skid control, cruise control, master warning triangle, position lights, amber "!" in a circle, amber oil level,
red/blue temperature-gauge zones, needles, orange LCD panels, glare spots.


Rules
1. A crop with no light in it and nothing cut off (reflections, glow, needles, gauge zones, display panels) -> `none`.
2. A crop with a light in it that is clipped, merged with other lights (giant box), or under about 25 px -> `finder_failures/`.
3. Right symbol, wrong color -> `none` until the manual says it is the same light.
4. Use the manual's color, not the photo's. Phone cameras turn overexposed red into orange.
5. Same function, different brand artwork (for example the word ABS or BRAKE instead of a circled symbol) -> same class.
6. Lit only. A light caught mid-fade still counts: keep it and add `_faded` to the file name.
7. Not sure? Put it in `to_check/` (never used for training or testing) and look it up in the owner's manual.
8. Label from the symbol and the manual, never from what the model said
9. Keep the car prefix in file names (mazda_, rav4_), plus night_ for the night session.