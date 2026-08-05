"""Vayal Nanban — a bilingual, farmer-first Tamil Nadu assistant."""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from farmer_assistant import (
    FarmerContext,
    LangSmithSettings,
    build_field_note,
    build_triage_card,
    generate_grounded_reply,
    offline_reply,
    transcribe_audio,
)
from rag_engine import build_rag_engine
from upload_safety import prepare_image_upload


load_dotenv()
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Vayal Nanban | வயல் நண்பன்",
    page_icon=":material/agriculture:",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo("assets/vayal-nanban-mark.png", size="large")


HERO_SLIDES = (
    {
        "src": "/app/static/hero/farmer-phone.png",
        "position": "68% center",
        "alt_en": "Tamil Nadu farmer checking crop guidance on a phone in a paddy field",
        "alt_ta": "நெல் வயலில் கைப்பேசியில் பயிர் வழிகாட்டலைப் பார்க்கும் தமிழ்நாடு விவசாயி",
        "caption_en": "Practical guidance for every field decision",
        "caption_ta": "ஒவ்வொரு வயல் முடிவுக்கும் நடைமுறை வழிகாட்டல்",
    },
    {
        "src": "/app/static/hero/smart-farm-analytics.png",
        "position": "50% center",
        "alt_en": "Paddy field with a tractor and digital crop analytics",
        "alt_ta": "டிராக்டர் மற்றும் டிஜிட்டல் பயிர் பகுப்பாய்வுடன் நெல் வயல்",
        "caption_en": "Farm context turned into clear next actions",
        "caption_ta": "வயல் தகவலிலிருந்து தெளிவான அடுத்த நடவடிக்கை",
    },
    {
        "src": "/app/static/hero/ai-farmer-companion.png",
        "position": "38% center",
        "alt_en": "Tamil farmer standing with a friendly AI assistant in a paddy field",
        "alt_ta": "நெல் வயலில் நட்பான AI உதவியாளருடன் நிற்கும் தமிழ் விவசாயி",
        "caption_en": "Technology that speaks like a trusted companion",
        "caption_ta": "நம்பகமான நண்பனைப் போல பேசும் தொழில்நுட்பம்",
    },
    {
        "src": "/app/static/hero/ai-field-guidance.png",
        "position": "35% center",
        "alt_en": "Tamil farmer receiving AI-supported field guidance",
        "alt_ta": "AI ஆதரவுடன் வயல் வழிகாட்டலைப் பெறும் தமிழ் விவசாயி",
        "caption_en": "Safer advice, at the moment it matters",
        "caption_ta": "தேவையான நேரத்தில் பாதுகாப்பான ஆலோசனை",
    },
)

_HERO_CAROUSEL = st.components.v2.component(
    "vayal_nanban_hero_carousel",
    html="""
<section class="vn-carousel" role="region" aria-roledescription="carousel" aria-label="Vayal Nanban highlights" tabindex="0">
  <div class="vn-slides" aria-live="off"></div>
  <div class="vn-shade" aria-hidden="true"></div>
  <div class="vn-copy">
    <span class="vn-eyebrow"></span>
    <p class="vn-caption"></p>
  </div>
  <button class="vn-arrow vn-prev" type="button" aria-label="Previous banner">&#8249;</button>
  <button class="vn-arrow vn-next" type="button" aria-label="Next banner">&#8250;</button>
  <div class="vn-dots" aria-label="Choose banner"></div>
  <button class="vn-toggle" type="button" aria-label="Pause slideshow">&#9208;</button>
</section>
""",
    css="""
:host {
  display: block;
  width: 100%;
  font-family: var(--st-font, sans-serif);
}

.vn-carousel {
  position: relative;
  width: 100%;
  height: clamp(210px, 30vw, 360px);
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--st-primary-color, #087a55) 28%, transparent);
  border-radius: var(--st-base-radius, 14px);
  background: #103a2b;
  box-shadow: 0 18px 46px rgba(18, 59, 44, 0.16);
  isolation: isolate;
  outline: none;
}

.vn-carousel:focus-visible {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--st-primary-color, #087a55) 32%, transparent),
              0 18px 46px rgba(18, 59, 44, 0.16);
}

.vn-slides,
.vn-slide,
.vn-slide img,
.vn-shade {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.vn-slide {
  margin: 0;
  opacity: 0;
  transform: scale(1.035);
  transition: opacity 780ms cubic-bezier(.22, .61, .36, 1), transform 3.8s ease;
  pointer-events: none;
}

.vn-slide.is-active {
  opacity: 1;
  transform: scale(1);
  z-index: 1;
}

.vn-slide img {
  display: block;
  object-fit: cover;
}

.vn-shade {
  z-index: 2;
  background: linear-gradient(90deg, rgba(8, 42, 30, .76) 0%, rgba(8, 42, 30, .23) 47%, transparent 76%),
              linear-gradient(0deg, rgba(5, 27, 20, .42) 0%, transparent 44%);
  pointer-events: none;
}

.vn-copy {
  position: absolute;
  z-index: 3;
  left: clamp(18px, 3vw, 38px);
  bottom: clamp(24px, 4vw, 42px);
  max-width: min(520px, 65%);
  color: #fff;
  text-shadow: 0 2px 18px rgba(5, 27, 20, .5);
}

.vn-eyebrow {
  display: inline-flex;
  padding: 6px 11px;
  border: 1px solid rgba(255, 255, 255, .34);
  border-radius: 999px;
  background: rgba(8, 68, 47, .52);
  backdrop-filter: blur(10px);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .035em;
}

.vn-caption {
  margin: 10px 0 0;
  font-family: var(--st-heading-font, var(--st-font, sans-serif));
  font-size: clamp(18px, 2.15vw, 29px);
  font-weight: 750;
  line-height: 1.2;
}

.vn-arrow,
.vn-toggle,
.vn-dot {
  border: 1px solid rgba(255, 255, 255, .42);
  color: #fff;
  background: rgba(8, 50, 36, .48);
  backdrop-filter: blur(10px);
  cursor: pointer;
  transition: transform 180ms ease, background-color 180ms ease, opacity 180ms ease;
}

.vn-arrow {
  position: absolute;
  z-index: 4;
  top: 50%;
  width: 40px;
  height: 40px;
  border-radius: 999px;
  font-size: 30px;
  line-height: 34px;
  transform: translateY(-50%);
  opacity: .72;
}

.vn-prev { left: 14px; }
.vn-next { right: 14px; }

.vn-arrow:hover,
.vn-arrow:focus-visible,
.vn-toggle:hover,
.vn-toggle:focus-visible {
  background: rgba(8, 122, 85, .84);
  opacity: 1;
}

.vn-arrow:hover { transform: translateY(-50%) scale(1.06); }

.vn-dots {
  position: absolute;
  z-index: 4;
  left: 50%;
  bottom: 13px;
  display: flex;
  gap: 7px;
  transform: translateX(-50%);
}

.vn-dot {
  width: 8px;
  height: 8px;
  padding: 0;
  border-radius: 999px;
  opacity: .65;
}

.vn-dot.is-active {
  width: 24px;
  background: #fff;
  opacity: 1;
}

.vn-toggle {
  position: absolute;
  z-index: 4;
  right: 14px;
  bottom: 12px;
  width: 32px;
  height: 32px;
  border-radius: 999px;
  font-size: 13px;
}

@media (max-width: 640px) {
  .vn-carousel { height: 220px; }
  .vn-copy { left: 16px; bottom: 40px; max-width: 76%; }
  .vn-caption { font-size: 18px; }
  .vn-eyebrow { font-size: 10px; }
  .vn-arrow { width: 34px; height: 34px; font-size: 25px; line-height: 28px; }
  .vn-prev { left: 8px; }
  .vn-next { right: 8px; }
}

@media (prefers-reduced-motion: reduce) {
  .vn-slide,
  .vn-arrow,
  .vn-toggle,
  .vn-dot { transition: none; }
}
""",
    js="""
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.vn-carousel');
  const slidesHost = parentElement.querySelector('.vn-slides');
  const dotsHost = parentElement.querySelector('.vn-dots');
  const caption = parentElement.querySelector('.vn-caption');
  const eyebrow = parentElement.querySelector('.vn-eyebrow');
  const previous = parentElement.querySelector('.vn-prev');
  const next = parentElement.querySelector('.vn-next');
  const toggle = parentElement.querySelector('.vn-toggle');
  const slidesData = Array.isArray(data?.slides) ? data.slides : [];
  const intervalMs = Number(data?.interval_ms) || 3000;
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  if (!root || !slidesHost || !dotsHost || !slidesData.length) return;

  slidesHost.replaceChildren();
  dotsHost.replaceChildren();
  eyebrow.textContent = data?.eyebrow || 'Vayal Nanban';

  const slideElements = slidesData.map((item, slideIndex) => {
    const figure = document.createElement('figure');
    figure.className = 'vn-slide';
    figure.setAttribute('aria-hidden', slideIndex === 0 ? 'false' : 'true');

    const image = document.createElement('img');
    image.src = item.src;
    image.alt = item.alt || '';
    image.loading = slideIndex === 0 ? 'eager' : 'lazy';
    image.decoding = 'async';
    image.draggable = false;
    image.style.objectPosition = item.position || 'center';
    figure.appendChild(image);
    slidesHost.appendChild(figure);
    return figure;
  });

  const dotElements = slidesData.map((item, dotIndex) => {
    const dot = document.createElement('button');
    dot.className = 'vn-dot';
    dot.type = 'button';
    dot.setAttribute('aria-label', `Show banner ${dotIndex + 1}`);
    dotsHost.appendChild(dot);
    return dot;
  });

  let current = 0;
  let timer = null;
  let manuallyPaused = false;
  let hoverPaused = false;
  let focusPaused = false;

  function show(index) {
    current = (index + slideElements.length) % slideElements.length;
    slideElements.forEach((slide, itemIndex) => {
      const active = itemIndex === current;
      slide.classList.toggle('is-active', active);
      slide.setAttribute('aria-hidden', active ? 'false' : 'true');
      dotElements[itemIndex].classList.toggle('is-active', active);
      dotElements[itemIndex].setAttribute('aria-current', active ? 'true' : 'false');
    });
    caption.textContent = slidesData[current].caption || '';
  }

  function shouldPause() {
    return manuallyPaused || hoverPaused || focusPaused || document.hidden || motionQuery.matches;
  }

  function syncTimer() {
    if (timer) window.clearInterval(timer);
    timer = null;
    if (!shouldPause()) {
      timer = window.setInterval(() => show(current + 1), intervalMs);
    }
    toggle.disabled = motionQuery.matches;
    toggle.textContent = manuallyPaused || motionQuery.matches ? '\u25B6' : '\u23F8';
    toggle.setAttribute(
      'aria-label',
      motionQuery.matches
        ? 'Slideshow paused for reduced motion'
        : (manuallyPaused ? 'Play slideshow' : 'Pause slideshow')
    );
  }

  function goTo(index) {
    show(index);
    syncTimer();
  }

  const onPrevious = () => goTo(current - 1);
  const onNext = () => goTo(current + 1);
  const onToggle = () => {
    if (motionQuery.matches) return;
    manuallyPaused = !manuallyPaused;
    syncTimer();
  };
  const onMouseEnter = () => { hoverPaused = true; syncTimer(); };
  const onMouseLeave = () => { hoverPaused = false; syncTimer(); };
  const onFocusIn = () => { focusPaused = true; syncTimer(); };
  const onFocusOut = (event) => {
    if (!root.contains(event.relatedTarget)) {
      focusPaused = false;
      syncTimer();
    }
  };
  const onKeyDown = (event) => {
    if (event.key === 'ArrowLeft') { event.preventDefault(); onPrevious(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); onNext(); }
  };
  const onVisibilityChange = () => syncTimer();
  const onMotionPreferenceChange = () => syncTimer();

  previous.addEventListener('click', onPrevious);
  next.addEventListener('click', onNext);
  toggle.addEventListener('click', onToggle);
  root.addEventListener('mouseenter', onMouseEnter);
  root.addEventListener('mouseleave', onMouseLeave);
  root.addEventListener('focusin', onFocusIn);
  root.addEventListener('focusout', onFocusOut);
  root.addEventListener('keydown', onKeyDown);
  document.addEventListener('visibilitychange', onVisibilityChange);
  motionQuery.addEventListener?.('change', onMotionPreferenceChange);
  dotElements.forEach((dot, dotIndex) => {
    dot.onclick = () => goTo(dotIndex);
  });

  show(0);
  syncTimer();

  return () => {
    if (timer) window.clearInterval(timer);
    previous.removeEventListener('click', onPrevious);
    next.removeEventListener('click', onNext);
    toggle.removeEventListener('click', onToggle);
    root.removeEventListener('mouseenter', onMouseEnter);
    root.removeEventListener('mouseleave', onMouseLeave);
    root.removeEventListener('focusin', onFocusIn);
    root.removeEventListener('focusout', onFocusOut);
    root.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    motionQuery.removeEventListener?.('change', onMotionPreferenceChange);
  };
}
""",
)


def render_hero_carousel(language: str) -> None:
    tamil = language != "English"
    slides = [
        {
            "src": slide["src"],
            "position": slide["position"],
            "alt": slide["alt_ta"] if tamil else slide["alt_en"],
            "caption": slide["caption_ta"] if tamil else slide["caption_en"],
        }
        for slide in HERO_SLIDES
    ]
    _HERO_CAROUSEL(
        key="hero_carousel",
        data={
            "slides": slides,
            "interval_ms": 3000,
            "eyebrow": "வயல் நண்பன் • தமிழ்நாடு" if tamil else "Vayal Nanban • Tamil Nadu",
        },
        width="stretch",
        height="content",
    )


DISTRICTS = [
    "Ariyalur",
    "Chengalpattu",
    "Chennai",
    "Coimbatore",
    "Cuddalore",
    "Dharmapuri",
    "Dindigul",
    "Erode",
    "Kallakurichi",
    "Kancheepuram",
    "Kanniyakumari",
    "Karur",
    "Krishnagiri",
    "Madurai",
    "Mayiladuthurai",
    "Nagapattinam",
    "Namakkal",
    "Nilgiris",
    "Perambalur",
    "Pudukkottai",
    "Ramanathapuram",
    "Ranipet",
    "Salem",
    "Sivaganga",
    "Tenkasi",
    "Thanjavur",
    "Theni",
    "Thoothukudi",
    "Tiruchirappalli",
    "Tirunelveli",
    "Tirupathur",
    "Tiruppur",
    "Tiruvallur",
    "Tiruvannamalai",
    "Tiruvarur",
    "Vellore",
    "Viluppuram",
    "Virudhunagar",
]

CROPS = [
    "Paddy / நெல்",
    "Sugarcane / கரும்பு",
    "Banana / வாழை",
    "Coconut / தென்னை",
    "Cotton / பருத்தி",
    "Groundnut / நிலக்கடலை",
    "Maize / மக்காச்சோளம்",
    "Millets / சிறுதானியம்",
    "Pulses / பயறு",
    "Tomato / தக்காளி",
    "Onion / வெங்காயம்",
    "Turmeric / மஞ்சள்",
    "Other / மற்றவை",
]

STAGES = [
    "Planning / திட்டமிடல்",
    "Sowing / விதைப்பு",
    "Seedling / நாற்று",
    "Vegetative / வளர்ச்சி",
    "Flowering / பூக்கும் நிலை",
    "Fruit or grain filling / காய் அல்லது மணி பிடித்தல்",
    "Harvest ready / அறுவடை நிலை",
]

IRRIGATION_TYPES = [
    "Rainfed / மானாவாரி",
    "Canal / கால்வாய்",
    "Borewell / ஆழ்துளை கிணறு",
    "Drip / சொட்டு நீர்",
    "Sprinkler / தெளிப்பு நீர்",
    "Other / மற்றவை",
]

COPY = {
    "தமிழ்": {
        "title": "வயல் நண்பன்",
        "tagline": "தமிழ்நாடு விவசாயிகளுக்கான எளிய, பாதுகாப்பான AI உதவியாளர்",
        "language": "மொழி",
        "district": "உங்கள் மாவட்டம்",
        "crop": "முக்கிய பயிர்",
        "stage": "பயிர் நிலை",
        "irrigation": "பாசன முறை",
        "context": "உங்கள் வயல் விவரம்",
        "new_chat": "புதிய உரையாடல்",
        "ai_on": "RAG + AI இணைக்கப்பட்டுள்ளது",
        "rag_ready": "அதிகாரப்பூர்வ RAG தயார்",
        "rag_caption": "FAISS தேடல் மூலம் தொடர்புடைய அதிகாரப்பூர்வ தகவல் மீட்டெடுக்கப்படும்.",
        "rag_unavailable": "RAG தேடல் கிடைக்கவில்லை; பாதுகாப்பான AI வழிகாட்டலை மட்டும் பயன்படுத்துகிறேன்.",
        "rag_sources": "ஆதாரத் துணுக்குகள் மீட்டெடுக்கப்பட்டன",
        "offline": "அடிப்படை வழிகாட்டல்",
        "online_caption": "ஆதாரத்துடன் விரிவான பதில் மற்றும் பட ஆய்வு கிடைக்கும்.",
        "offline_caption": "API key இல்லாமலும் பொதுவான வழிகாட்டல் கிடைக்கும்.",
        "welcome": "வணக்கம்! இன்று உங்கள் வயலில் என்ன உதவி வேண்டும்?",
        "welcome_caption": "கேள்வியை தட்டச்சு செய்யுங்கள், குரலில் பதிவு செய்யுங்கள், அல்லது பயிர் படத்தை இணைக்குங்கள்.",
        "ask": "உங்கள் விவசாயக் கேள்வியை கேளுங்கள்…",
        "try": "விரைவாக கேளுங்கள்",
        "voice_missing": "குரல் பதிவு பெறப்பட்டது. அதை எழுத்தாக மாற்ற AI key தேவை; கேள்வியை தட்டச்சு செய்யவும்.",
        "image_prompt": "இந்த பயிர் படத்தை கவனமாக பார்த்து, சாத்தியமான பிரச்சனைகளையும் அடுத்த பாதுகாப்பான நடவடிக்கைகளையும் கூறுங்கள்.",
        "bad_image": "இந்த கோப்பை பாதுகாப்பான பயிர் படமாக திறக்க முடியவில்லை. JPG, PNG அல்லது WebP படத்தை மீண்டும் இணைக்கவும்.",
        "error": "AI சேவை இப்போது கிடைக்கவில்லை. அதனால் பாதுகாப்பான அடிப்படை வழிகாட்டலை காட்டுகிறேன்.",
        "sources": "அதிகாரப்பூர்வ தகவல்",
        "safety": "மருந்து அளவு அல்லது கலவை குறித்து லேபிள் மற்றும் வேளாண்மை நிபுணரின் அறிவுரையை உறுதி செய்யுங்கள்.",
        "privacy": "படங்களில் தனிப்பட்ட தகவலை இணைக்க வேண்டாம். AI/LangSmith இணைக்கப்பட்டால் படம் மற்றும் கேள்வி அந்த சேவைகளுக்கு அனுப்பப்படலாம்.",
        "footer": "வயல் நண்பன் வழங்குவது பொதுவான வழிகாட்டல்; இது நேரடி கள ஆய்வுக்கு மாற்றாகாது.",
        "farm_snapshot": "உங்கள் வயல் சுருக்கம்",
        "action_card": "வயல் செயல் அட்டை",
        "today": "இன்று செய்யுங்கள்",
        "watch": "அடுத்து கவனியுங்கள்",
        "escalate": "உதவி பெற வேண்டிய நேரம்",
        "official_source": "அதிகாரப்பூர்வ ஆதாரம்",
        "download_note": "களக் குறிப்பை பதிவிறக்கவும்",
        "feedback": "இந்த பதில் உதவியதா?",
        "feedback_saved": "நன்றி — உங்கள் கருத்து இந்த உரையாடலில் சேமிக்கப்பட்டது.",
        "features": (
            (":material/health_and_safety:", "பாதிப்பு முன்னுரிமை", "கேள்வி எவ்வளவு அவசரம் என்பதை தெளிவாகக் காட்டும்."),
            (":material/library_books:", "ஆதார RAG", "அதிகாரப்பூர்வ தகவலைத் தேடி ஆதார இணைப்புடன் பதிலளிக்கும்."),
            (":material/share:", "பகிரக்கூடிய குறிப்பு", "வேளாண்மை அலுவலரிடம் காட்ட களக் குறிப்பை சேமிக்கலாம்."),
        ),
        "suggestions": {
            ":material/pest_control: இலை நோயை கண்டறிய உதவுங்கள்": "என் பயிரின் இலைகளில் புள்ளிகள் உள்ளன. என்ன விவரங்கள் மற்றும் படம் தேவை?",
            ":material/water_drop: பாசன திட்டம் வேண்டும்": "என் பயிருக்கான பாதுகாப்பான பாசன இடைவெளியை திட்டமிட உதவுங்கள்.",
            ":material/payments: சந்தை விலை ஒப்பீடு": "என் விளைபொருளை விற்க இரண்டு சந்தைகளை எப்படி ஒப்பிடுவது?",
            ":material/account_balance: மானிய திட்டம் தேடுங்கள்": "எனக்கு பொருந்தும் வேளாண் மானிய திட்டத்தை கண்டறிய என்ன விவரங்கள் தேவை?",
        },
    },
    "English": {
        "title": "Vayal Nanban",
        "tagline": "A simple, safe AI companion for Tamil Nadu farmers",
        "language": "Language",
        "district": "Your district",
        "crop": "Main crop",
        "stage": "Crop stage",
        "irrigation": "Irrigation method",
        "context": "Your farm context",
        "new_chat": "New conversation",
        "ai_on": "RAG + AI connected",
        "rag_ready": "Official-source RAG ready",
        "rag_caption": "FAISS retrieves relevant passages from the verified agriculture knowledge base.",
        "rag_unavailable": "RAG retrieval is unavailable, so I am using the safety-focused AI guidance only.",
        "rag_sources": "source passages retrieved",
        "offline": "Essential guidance",
        "online_caption": "Grounded answers, citations, and image guidance are available.",
        "offline_caption": "General guidance works even without an API key.",
        "welcome": "Vanakkam! What can I help with on your farm today?",
        "welcome_caption": "Type a question, record your voice, or attach a crop photo.",
        "ask": "Ask your farming question…",
        "try": "Quick questions",
        "voice_missing": "Voice received. An AI key is needed to transcribe it; please type the question for now.",
        "image_prompt": "Review this crop photo carefully and explain possible causes and the next safe actions.",
        "bad_image": "I could not open that as a safe crop photo. Please attach a valid JPG, PNG, or WebP image.",
        "error": "The AI service is unavailable right now, so I am showing safe essential guidance instead.",
        "sources": "Official information",
        "safety": "Confirm pesticide doses and combinations using the product label and a qualified agriculture expert.",
        "privacy": "Do not attach personal information. When AI/LangSmith is connected, the photo and question may be sent to those services.",
        "footer": "Vayal Nanban provides general guidance and does not replace a field inspection.",
        "farm_snapshot": "Your farm snapshot",
        "action_card": "Field action card",
        "today": "Do today",
        "watch": "Watch next",
        "escalate": "Get help when",
        "official_source": "Official source",
        "download_note": "Download field note",
        "feedback": "Was this useful?",
        "feedback_saved": "Thank you — your rating is saved in this conversation.",
        "features": (
            (":material/health_and_safety:", "Smart triage", "Shows how urgently the question needs attention."),
            (":material/library_books:", "Grounded RAG", "Retrieves verified passages and links every answer to its sources."),
            (":material/share:", "Shareable note", "Save a clear field note to show an agriculture officer."),
        ),
        "suggestions": {
            ":material/pest_control: Diagnose a leaf problem": "My crop has spots on the leaves. What details and photos do you need?",
            ":material/water_drop: Plan irrigation": "Help me plan a safe irrigation interval for my crop.",
            ":material/payments: Compare market prices": "How should I compare two markets before selling my produce?",
            ":material/account_balance: Find a subsidy": "What details do you need to identify a suitable agriculture subsidy scheme?",
        },
    },
}


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return os.getenv(name, default)


def get_bool_secret(name: str, default: bool = False) -> bool:
    value = get_secret(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


@st.cache_resource(show_spinner=False)
def get_cached_rag_engine(_api_key: str, embedding_model: str):
    """Build the FAISS index once per Streamlit process."""

    return build_rag_engine(api_key=_api_key, embedding_model=embedding_model)


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_prompt", None)


def render_action_card(
    card: dict[str, str],
    field_note: str,
    labels: dict[str, Any],
    key_suffix: str,
) -> None:
    color = {"urgent": "red", "attention": "orange", "routine": "green"}[card["level"]]
    icon = {
        "urgent": ":material/emergency:",
        "attention": ":material/priority_high:",
        "routine": ":material/check_circle:",
    }[card["level"]]

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f":material/checklist: {labels['action_card']}")
            st.badge(card["label"], icon=icon, color=color)

        action_columns = st.columns(3)
        action_items = (
            (labels["today"], ":material/today:", card["today"]),
            (labels["watch"], ":material/visibility:", card["watch"]),
            (labels["escalate"], ":material/support_agent:", card["escalate"]),
        )
        for column, (heading, item_icon, body) in zip(action_columns, action_items):
            with column.container(border=True, height="stretch"):
                st.markdown(f"**{item_icon} {heading}**")
                st.caption(body)

        with st.container(horizontal=True):
            st.link_button(
                labels["official_source"],
                card["source_url"],
                icon=":material/verified:",
            )
            st.download_button(
                labels["download_note"],
                data=field_note,
                file_name=f"vayal-nanban-{key_suffix}.md",
                mime="text/markdown",
                key=f"field_note_{key_suffix}",
                icon=":material/download:",
                on_click="ignore",
            )


st.session_state.setdefault("messages", [])
st.session_state.setdefault("language", "தமிழ்")
st.session_state.setdefault("district", "Thanjavur")
st.session_state.setdefault("crop", "Paddy / நெல்")
st.session_state.setdefault("stage", "Vegetative / வளர்ச்சி")
st.session_state.setdefault("irrigation", "Canal / கால்வாய்")

api_key = get_secret("OPENAI_API_KEY")
model = get_secret("OPENAI_MODEL", "gpt-4.1-mini")
embedding_model = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
langsmith_settings = LangSmithSettings(
    enabled=get_bool_secret("LANGSMITH_TRACING"),
    api_key=get_secret("LANGSMITH_API_KEY"),
    project=get_secret("LANGSMITH_PROJECT", "vayal-nanban-buildthon"),
    endpoint=get_secret("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    workspace_id=get_secret("LANGSMITH_WORKSPACE_ID"),
)

with st.sidebar:
    st.subheader(":material/agriculture: Vayal Nanban")
    language = st.segmented_control(
        "Language / மொழி",
        ["தமிழ்", "English"],
        key="language",
        required=True,
        width="stretch",
    )
    t = COPY[language]

    st.caption(t["context"])
    district = st.selectbox(t["district"], DISTRICTS, key="district")
    crop = st.selectbox(t["crop"], CROPS, key="crop")
    stage = st.selectbox(t["stage"], STAGES, key="stage")
    irrigation = st.selectbox(t["irrigation"], IRRIGATION_TYPES, key="irrigation")

    if api_key:
        st.badge(t["rag_ready"], icon=":material/library_books:", color="green")
        st.caption(t["rag_caption"])
        st.badge(t["ai_on"], icon=":material/check_circle:", color="green")
        st.caption(t["online_caption"])
    else:
        st.badge(t["offline"], icon=":material/offline_bolt:", color="orange")
        st.caption(t["offline_caption"])

    st.button(
        t["new_chat"],
        icon=":material/add_comment:",
        on_click=reset_conversation,
        width="stretch",
    )

    with st.expander(t["sources"], icon=":material/verified:"):
        st.link_button(
            "TNAU Agritech",
            "https://agritech.tnau.ac.in/",
            icon=":material/menu_book:",
            width="stretch",
        )
        st.link_button(
            "IMD Chennai",
            "https://mausam.imd.gov.in/chennai/",
            icon=":material/cloud:",
            width="stretch",
        )
        st.link_button(
            "TN Agriculture",
            "https://www.tnagrisnet.tn.gov.in/",
            icon=":material/account_balance:",
            width="stretch",
        )
        st.link_button(
            "TN Agri Market",
            "https://www.agrimark.tn.gov.in/",
            icon=":material/storefront:",
            width="stretch",
        )

    st.warning(t["safety"], icon=":material/health_and_safety:")
    st.caption(t["privacy"])

context = FarmerContext(
    language=language,
    district=district,
    crop=crop,
    stage=stage,
    irrigation=irrigation,
)

with st.container(border=True, key="hero"):
    render_hero_carousel(language)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.image("assets/vayal-nanban-mark.png", width=68)
        st.title(t["title"])
        if api_key:
            st.badge(t["ai_on"], icon=":material/auto_awesome:", color="green")
        else:
            st.badge(t["offline"], icon=":material/offline_bolt:", color="orange")
    st.markdown(f"### {t['tagline']}")
    st.caption(f":material/location_on: {district}  ·  :material/eco: {crop}")
    st.caption(t["farm_snapshot"])
    snapshot_columns = st.columns(3)
    snapshot_items = (
        (":material/grass:", t["crop"], crop),
        (":material/monitoring:", t["stage"], stage),
        (":material/water_drop:", t["irrigation"], irrigation),
    )
    for column, (icon, label, value) in zip(snapshot_columns, snapshot_items):
        with column:
            st.caption(f"{icon} {label}")
            st.markdown(f"**{value}**")

pending_prompt = st.session_state.pop("pending_prompt", None)

if not st.session_state.messages and not pending_prompt:
    st.space("small")
    st.subheader(t["welcome"])
    st.caption(t["welcome_caption"])

    feature_columns = st.columns(3)
    for column, (icon, heading, caption) in zip(feature_columns, t["features"]):
        with column.container(border=True, height="stretch"):
            st.subheader(f"{icon} {heading}")
            st.caption(caption)

    selected = st.pills(
        t["try"],
        list(t["suggestions"].keys()),
        key="starter_question",
        width="stretch",
    )
    if selected:
        st.session_state.pending_prompt = t["suggestions"][selected]
        st.rerun()

for message_index, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "assets/vayal-nanban-mark.png" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        for attachment in message.get("images", []):
            try:
                st.image(attachment["data"], caption=attachment["name"], width=360)
            except Exception:
                LOGGER.exception("A stored chat image could not be rendered")
                st.warning(t["bad_image"])
        if role == "assistant" and message.get("triage"):
            if message.get("rag_sources"):
                st.badge(
                    f"{len(message['rag_sources'])} {t['rag_sources']}",
                    icon=":material/library_books:",
                    color="green",
                )
            render_action_card(
                message["triage"],
                message.get("field_note", message["content"]),
                t,
                str(message_index),
            )
            st.caption(t["feedback"])
            rating = st.feedback("thumbs", key=f"feedback_{message_index}")
            if rating is not None:
                message["feedback"] = rating
                st.caption(t["feedback_saved"])

submission = st.chat_input(
    t["ask"],
    key="farmer_chat_input",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
    accept_audio=True,
    max_chars=1200,
    max_upload_size=8,
    submit_mode="disable",
)

prompt_text = pending_prompt or ""
images: list[dict[str, Any]] = []
audio = None

if submission:
    prompt_text = submission.text.strip()
    audio = submission.audio
    for uploaded in submission.files:
        try:
            images.append(prepare_image_upload(uploaded.name, uploaded.getvalue()))
        except ValueError:
            LOGGER.warning("Rejected an unreadable or unsupported image upload")
            st.warning(t["bad_image"])

if (pending_prompt or submission) and audio and not prompt_text:
    if api_key:
        try:
            with st.spinner("குரலை எழுத்தாக மாற்றுகிறேன்…" if language == "தமிழ்" else "Transcribing voice…"):
                prompt_text = transcribe_audio(api_key, audio, language)
        except Exception:
            LOGGER.exception("Voice transcription failed")
            prompt_text = t["voice_missing"]
    else:
        prompt_text = t["voice_missing"]

if (pending_prompt or submission) and not prompt_text and images:
    prompt_text = t["image_prompt"]

if (pending_prompt or submission) and prompt_text:
    user_message = {"role": "user", "content": prompt_text, "images": images}
    st.session_state.messages.append(user_message)
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt_text)
        for attachment in images:
            try:
                st.image(attachment["data"], caption=attachment["name"], width=360)
            except Exception:
                LOGGER.exception("A validated image could not be rendered")
                st.warning(t["bad_image"])
        if audio:
            st.audio(audio)

    grounded_reply = None
    with st.chat_message("assistant", avatar="assets/vayal-nanban-mark.png"):
        try:
            with st.skeleton(height=110):
                if api_key:
                    rag_engine = None
                    try:
                        rag_engine = get_cached_rag_engine(api_key, embedding_model)
                    except Exception:
                        LOGGER.exception("RAG index creation failed; continuing without retrieval")
                        st.caption(t["rag_unavailable"])

                    grounded_reply = generate_grounded_reply(
                        api_key=api_key,
                        model=model,
                        history=st.session_state.messages,
                        context=context,
                        image=images[0] if images else None,
                        langsmith=langsmith_settings,
                        rag_engine=rag_engine,
                    )
                    answer = grounded_reply.text
                else:
                    answer = offline_reply(prompt_text, context)
                    grounded_reply = None
        except Exception:
            LOGGER.exception("AI answer generation failed; serving offline guidance")
            st.caption(t["error"])
            answer = offline_reply(prompt_text, context)
        st.markdown(answer)

    triage = build_triage_card(prompt_text, context)
    field_note = build_field_note(prompt_text, answer, triage, context)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "triage": triage,
            "field_note": field_note,
            "rag_sources": [source.as_dict() for source in grounded_reply.sources]
            if grounded_reply
            else [],
        }
    )
    st.rerun()

st.caption(t["footer"], text_alignment="center")

