import os
import unittest
from unittest.mock import patch

from normalizers.classic_to_resume_schema_v1 import classic_sections_to_resume_schema_v1
from schemas.resume_schema_defaults import apply_resume_schema_v1_defaults
from src.utils.parser_orchestrator import parse_resume_to_schema_v1
from src.utils.resume_schema_validator import validate_resume_schema_v1


class TestResumeSchemaV1Flow(unittest.TestCase):
    def test_defaults_populate_required_schema_keys(self):
        resume = apply_resume_schema_v1_defaults({"summary": "Experienced engineer"})

        self.assertEqual(resume["personal_information"], {})
        self.assertEqual(resume["summary"], "Experienced engineer")
        self.assertEqual(resume["work_experience"], [])
        self.assertEqual(resume["education"], [])
        self.assertEqual(resume["skills"], {})
        self.assertEqual(resume["projects"], [])
        self.assertEqual(resume["volunteer_experience"], [])
        self.assertEqual(resume["custom_sections"], [])

    def test_classic_sections_map_to_resume_schema_v1(self):
        resume = classic_sections_to_resume_schema_v1(
            {
                "name": ("Jane Doe", 0.8),
                "skills": ("Python, SQL\nLeadership", 0.7),
                "experience": ("Developer at Example Co", 0.6),
                "certifications": ("AWS Certified", 0.6),
            },
            raw_text="Jane Doe jane@example.com +91 9876543210",
        )

        self.assertEqual(resume["personal_information"]["name"], "Jane Doe")
        self.assertEqual(resume["personal_information"]["email"], "jane@example.com")
        self.assertEqual(resume["skills"]["items"], ["Python", "SQL", "Leadership"])
        self.assertEqual(resume["work_experience"], [{"description": "Developer at Example Co"}])
        self.assertEqual(resume["custom_sections"][0]["section_title"], "Certifications")

    def test_orchestrator_uses_classic_when_ai_disabled(self):
        def classic_extractor(text, file_path, filename):
            return {"name": ("Jane Doe", 0.8), "skills": ("Python", 0.7)}

        with patch.dict(os.environ, {"AI_ENABLED": "false"}, clear=False):
            result = parse_resume_to_schema_v1(
                text="Jane Doe jane@example.com",
                filename="resume.txt",
                file_path="resume.txt",
                classic_extractor=classic_extractor,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["parser_mode"], "classic")
        self.assertFalse(result["fallback_used"])
        self.assertTrue(result["validation"]["is_valid"])

    def test_validator_rejects_non_object_array_items(self):
        validation = validate_resume_schema_v1({"work_experience": ["not object"]})

        self.assertFalse(validation["is_valid"])
        self.assertIn("'work_experience[0]' must be object", validation["errors"])


if __name__ == "__main__":
    unittest.main()
