import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from farmer_assistant import (
    FarmerContext,
    LangSmithSettings,
    build_field_note,
    build_system_prompt,
    build_trace_config,
    build_triage_card,
    detect_topic,
    generate_ai_reply,
    offline_reply,
)


class FarmerAssistantTests(unittest.TestCase):
    def setUp(self):
        self.english = FarmerContext("English", "Thanjavur", "Paddy / நெல்")
        self.tamil = FarmerContext("தமிழ்", "Erode", "Turmeric / மஞ்சள்")

    def test_detects_english_and_tamil_topics(self):
        self.assertEqual(detect_topic("Will it rain tomorrow?"), "weather")
        self.assertEqual(detect_topic("இலையில் பூச்சி இருக்கிறது"), "pest")
        self.assertEqual(detect_topic("மானியம் எப்படி பெறுவது?"), "scheme")

    def test_offline_reply_keeps_farm_context(self):
        reply = offline_reply("How should I irrigate?", self.english)
        self.assertIn("Thanjavur", reply)
        self.assertIn("Paddy", reply)

    def test_tamil_reply_is_localized(self):
        reply = offline_reply("மண் பரிசோதனை", self.tamil)
        self.assertIn("மண் பரிசோதனை", reply)
        self.assertIn("Erode", reply)

    def test_system_prompt_has_safety_boundaries(self):
        prompt = build_system_prompt(self.english)
        self.assertIn("Never invent live weather", prompt)
        self.assertIn("Never guess pesticide", prompt)
        self.assertIn("Thanjavur", prompt)

    def test_pest_question_gets_attention_action_card(self):
        card = build_triage_card("The leaf spots are spreading rapidly", self.english)
        self.assertEqual(card["level"], "attention")
        self.assertIn("TNAU", card["source_name"])
        self.assertTrue(card["today"])

    def test_safety_hazard_gets_urgent_triage(self):
        card = build_triage_card("A worker feels dizzy after possible poisoning", self.english)
        self.assertEqual(card["level"], "urgent")
        self.assertIn("112", card["escalate"])
        self.assertEqual(card["source_url"], "https://112.gov.in/")

    def test_realistic_pesticide_exposure_phrasings_are_urgent(self):
        questions = (
            "A worker feels dizzy after spraying pesticide. What should we do?",
            "worker fainted after spraying",
            "he is vomiting after the spray",
            "worker feels dizzy and nauseous",
            "chemical went into his eyes",
            "worker swallowed pesticide",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(build_triage_card(question, self.english)["level"], "urgent")

    def test_ordinary_tamil_words_do_not_trigger_an_emergency(self):
        questions = (
            "இந்த பிரச்சனைக்கு தீர்வு என்ன?",
            "கால்நடை தீவனம் பற்றி சொல்லுங்கள்",
            "நோய் தீவிரம் அதிகமா?",
            "விதைப்புக்கு தீர்மானம் எடுக்கவா?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertNotEqual(build_triage_card(question, self.tamil)["level"], "urgent")

    def test_field_note_contains_farm_context_and_actions(self):
        card = build_triage_card("How should I irrigate?", self.english)
        note = build_field_note("How should I irrigate?", "Check moisture.", card, self.english)
        self.assertIn("Thanjavur", note)
        self.assertIn("## Action card", note)
        self.assertIn(card["source_url"], note)

    def test_langsmith_defaults_are_safe_and_disabled(self):
        settings = LangSmithSettings()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.api_key, "")
        self.assertEqual(settings.project, "vayal-nanban-buildthon")

    def test_trace_config_groups_runs_without_personal_identifiers(self):
        config = build_trace_config(self.english)
        self.assertEqual(config["run_name"], "vayal_nanban_answer")
        self.assertIn("buildthon", config["tags"])
        self.assertIn("english", config["tags"])
        self.assertEqual(config["metadata"]["district"], "Thanjavur")
        self.assertEqual(config["metadata"]["crop"], self.english.crop)
        self.assertNotIn("user_id", config["metadata"])
        self.assertNotIn("session_id", config["metadata"])

    @patch("farmer_assistant._get_langsmith_client")
    @patch("farmer_assistant.tracing_context")
    @patch("farmer_assistant.ChatOpenAI")
    def test_langsmith_stays_off_when_key_is_missing(self, chat_openai, tracing, get_client):
        chat_openai.return_value.invoke.return_value = AIMessage(content="Safe answer")
        answer = generate_ai_reply(
            api_key="openai-test-key",
            model="test-model",
            history=[{"role": "user", "content": "How should I irrigate?"}],
            context=self.english,
            langsmith=LangSmithSettings(enabled=True, api_key=""),
        )
        self.assertEqual(answer, "Safe answer")
        get_client.assert_not_called()
        tracing.assert_not_called()

    @patch("farmer_assistant._get_langsmith_client")
    @patch("farmer_assistant.tracing_context")
    @patch("farmer_assistant.ChatOpenAI")
    def test_langsmith_enters_tracing_context_when_configured(self, chat_openai, tracing, get_client):
        chat_openai.return_value.invoke.return_value = AIMessage(content="Traced answer")
        tracing.return_value = MagicMock()
        answer = generate_ai_reply(
            api_key="openai-test-key",
            model="test-model",
            history=[{"role": "user", "content": "How should I irrigate?"}],
            context=self.english,
            langsmith=LangSmithSettings(enabled=True, api_key="langsmith-test-key"),
        )
        self.assertEqual(answer, "Traced answer")
        get_client.assert_called_once()
        tracing.assert_called_once()


if __name__ == "__main__":
    unittest.main()

