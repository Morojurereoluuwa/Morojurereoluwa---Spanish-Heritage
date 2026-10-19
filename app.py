import html
import random
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Hello World Motion Lab", page_icon="✨", layout="wide")

st.title("✨ Hello World Motion Lab")
st.caption("A tiny app-building playground: type a name, style it, and bring it to life.")

with st.sidebar:
    st.header("Customize your app")
    name = st.text_input("Your name or message", value="Hello World", max_chars=100)

    font_family = st.selectbox(
        "Font type",
        ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Trebuchet MS", "Impact", "Comic Sans MS"],
    )
    font_size = st.slider("Font size", min_value=20, max_value=100, value=52, step=2)

    background = st.selectbox(
        "Background",
        ["Soft Sky", "Midnight", "Sunset", "Forest", "Lavender", "Paper"],
    )
    name_mode = st.selectbox("Name display", ["Centered", "Smooth ticker", "Repeated pattern"])
    effect = st.selectbox(
        "Dynamic action",
        ["None", "Rain", "Snow", "Hail", "Wind gusts", "Confetti", "Tornado"],
    )
    speed = st.select_slider("Animation speed", options=["Slow", "Normal", "Fast"], value="Normal")
    st.markdown("---")
    st.caption("Tip: change one setting at a time to see how it affects the result.")

backgrounds = {
    "Soft Sky": ("#dff4ff", "#17324d", "#b6e4fa"),
    "Midnight": ("#101426", "#f7f7ff", "#252d50"),
    "Sunset": ("#ffdfc7", "#542a52", "#ffb28d"),
    "Forest": ("#143b32", "#f3ffe8", "#235c49"),
    "Lavender": ("#eee5ff", "#34265c", "#d7c4ff"),
    "Paper": ("#fffdf5", "#242424", "#eee9dc"),
}
bg_color, text_color, accent_color = backgrounds[background]
durations = {"Slow": 1.6, "Normal": 1.0, "Fast": 0.55}
duration = durations[speed]

safe_name = html.escape(name.strip() or "Hello World")
safe_font = font_family.replace('"', "")
safe_name_for_attr = safe_name

# Create separate animated particles so effects loop continuously inside the embedded browser.
rng = random.Random(27)
particles = []
if effect in ("Rain", "Snow", "Hail"):
    symbol = {"Rain": "💧", "Snow": "❄", "Hail": "🧊"}[effect]
    for i in range(48 if effect == "Rain" else 34):
        left = rng.randint(0, 100)
        delay = rng.uniform(-8, 0)
        fall_duration = rng.uniform(1.2, 2.8) * duration
        size = rng.randint(10, 23) if effect != "Hail" else rng.randint(9, 17)
        particles.append(
            f'<span class="particle {effect.lower()}" style="left:{left}%;animation-delay:{delay:.2f}s;'
            f'animation-duration:{fall_duration:.2f}s;font-size:{size}px">{symbol}</span>'
        )
elif effect == "Confetti":
    colors = ["#ff477e", "#ffd166", "#06d6a0", "#4cc9f0", "#9b5de5", "#f77f00"]
    for i in range(65):
        left = rng.randint(0, 100)
        delay = rng.uniform(-7, 0)
        fall_duration = rng.uniform(2.2, 4.6) * duration
        width = rng.randint(5, 10)
        height = rng.randint(8, 15)
        color = rng.choice(colors)
        particles.append(
            f'<span class="confetti" style="left:{left}%;animation-delay:{delay:.2f}s;'
            f'animation-duration:{fall_duration:.2f}s;width:{width}px;height:{height}px;background:{color}"></span>'
        )
elif effect == "Wind gusts":
    for i in range(12):
        top = rng.randint(5, 95)
        delay = rng.uniform(-5, 0)
        particles.append(
            f'<span class="wind-line" style="top:{top}%;animation-delay:{delay:.2f}s;'
            f'animation-duration:{rng.uniform(1.8, 3.2) * duration:.2f}s"></span>'
        )
elif effect == "Tornado":
    for i in range(28):
        delay = rng.uniform(-4, 0)
        particles.append(
            f'<span class="tornado-bit" style="--i:{i};animation-delay:{delay:.2f}s;'
            f'animation-duration:{rng.uniform(1.8, 3.2) * duration:.2f}s">{"✦" if i % 2 else "•"}</span>'
        )

effect_markup = "".join(particles)
if name_mode == "Smooth ticker":
    name_markup = f'<div class="ticker"><div class="ticker-track"><span>{safe_name}</span><span>{safe_name}</span><span>{safe_name}</span></div></div>'
elif name_mode == "Repeated pattern":
    name_markup = '<div class="repeated">' + "".join(f"<span>{safe_name}</span>" for _ in range(6)) + "</div>"
else:
    name_markup = f'<div class="center-name">{safe_name}</div>'

effect_class = {
    "None": "no-effect",
    "Rain": "rain-scene",
    "Snow": "snow-scene",
    "Hail": "hail-scene",
    "Wind gusts": "wind-scene",
    "Confetti": "confetti-scene",
    "Tornado": "tornado-scene",
}[effect]

document = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
.scene {{
  position: relative; width: 100%; height: 560px; overflow: hidden;
  border-radius: 20px; background: {bg_color}; color: {text_color};
  font-family: "{safe_font}", sans-serif;
  box-shadow: inset 0 0 0 1px {accent_color};
}}
.effects {{ position:absolute; inset:0; overflow:hidden; pointer-events:none; }}
.name-layer {{
  position:absolute; inset:0; z-index:5; display:flex; align-items:center;
  justify-content:center; padding:32px; font-size:{font_size}px; font-weight:700;
  line-height:1.2; overflow:hidden; overflow-wrap:anywhere;
  text-shadow: 0 2px 14px rgba(0,0,0,.10);
}}
.center-name {{ text-align:center; max-width:100%; }}
.ticker {{ width:100%; overflow:hidden; white-space:nowrap; }}
.ticker-track {{ width:max-content; display:flex; gap:1.2em; animation: ticker {18 * duration:.2f}s linear infinite; }}
.ticker-track span {{ flex:0 0 auto; }}
.repeated {{ width:100%; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; text-align:center; }}
.particle {{ position:absolute; top:-40px; opacity:.88; animation: fall linear infinite; }}
.snow {{ filter:drop-shadow(0 0 4px rgba(255,255,255,.5)); }}
.hail {{ filter:drop-shadow(0 2px 3px rgba(0,0,0,.2)); }}
@keyframes fall {{ from {{ transform:translateY(-50px) rotate(0deg); }} to {{ transform:translateY(650px) rotate(360deg); }} }}
.confetti {{ position:absolute; top:-24px; border-radius:2px; animation: confetti-fall linear infinite; }}
@keyframes confetti-fall {{ 0% {{ transform:translateY(-30px) rotate(0deg); }} 100% {{ transform:translateY(620px) rotate(900deg); }} }}
.wind-line {{ position:absolute; left:-35%; width:30%; height:3px; border-radius:99px; opacity:.55;
  background:linear-gradient(90deg,transparent,{accent_color},transparent); animation: gust ease-in-out infinite; }}
.wind-line:nth-child(3n) {{ height:5px; }}
@keyframes gust {{ 0% {{ transform:translateX(0) skewX(-20deg); opacity:0; }} 20% {{ opacity:.8; }} 100% {{ transform:translateX(500%) skewX(-20deg); opacity:0; }} }}
.tornado-bit {{ position:absolute; left:50%; top:50%; color:{accent_color}; font-size:22px; opacity:.8;
  animation: swirl ease-in-out infinite; transform-origin:0 0; }}
@keyframes swirl {{
  0% {{ transform:rotate(0deg) translateX(calc(25px + var(--i) * 2px)) scale(.7); opacity:.2; }}
  50% {{ opacity:.9; }}
  100% {{ transform:rotate(360deg) translateX(calc(25px + var(--i) * 2px)) scale(1); opacity:.2; }}
}}
.no-effect {{ display:none; }}
.scene-footer {{ position:absolute; right:16px; bottom:12px; z-index:8; font:12px/1.3 Arial,sans-serif; opacity:.55; }}
@keyframes ticker {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-33.333%); }} }}
@media (prefers-reduced-motion: reduce) {{
  .particle,.confetti,.wind-line,.tornado-bit,.ticker-track {{ animation-duration:30s !important; }}
}}
</style>
</head>
<body>
<div class="scene">
  <div class="effects {effect_class}">{effect_markup}</div>
  <div class="name-layer">{name_markup}</div>
  <div class="scene-footer">HELLO WORLD MOTION LAB</div>
</div>
</body>
</html>
"""

components.html(document, height=580, scrolling=False)

st.markdown("### What you just built")
st.write(
    "This app combines **inputs** (your controls), **variables** (the selected settings), "
    "and **HTML/CSS** (the display and continuous animations). The animation runs in the browser, "
    "so the page does not need a Python rerun every fraction of a second."
)
