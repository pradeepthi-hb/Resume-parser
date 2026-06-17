import unittest

from normalizers.resume_schema_v1_section_mapper import resume_schema_v1_section_text
from normalizers.resume_schema_v1_to_builder import build_resume_builder_response


class TestResumeSchemaV1Adapters(unittest.TestCase):
    def setUp(self):
        self.resume = {
            "personal_information": {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "phone": "+91 9876543210",
            },
            "summary": "Backend engineer.",
            "work_experience": [
                {
                    "job_title": "Python Developer",
                    "company": "Example Labs",
                    "description": "Built APIs",
                }
            ],
            "education": [{"degree": "B.Tech", "institution": "ABC University"}],
            "skills": {"technical": ["Python", "SQL"], "soft_skills": ["Leadership"]},
            "projects": [{"title": "Resume Parser", "description": "AI parser"}],
            "volunteer_experience": [],
            "custom_sections": [{"section_title": "Certifications", "items": ["AWS Certified"]}],
        }

    def test_section_mapper_reads_personal_name(self):
        self.assertEqual(resume_schema_v1_section_text(self.resume, "name"), "Jane Doe")

    def test_section_mapper_flattens_skills(self):
        text = resume_schema_v1_section_text(self.resume, "skills")
        self.assertIn("Python", text)
        self.assertIn("SQL", text)
        self.assertIn("Leadership", text)

    def test_section_mapper_reads_custom_certifications(self):
        self.assertEqual(resume_schema_v1_section_text(self.resume, "certifications"), "AWS Certified")

    def test_builder_adapter_keeps_legacy_envelope_keys(self):
        response = build_resume_builder_response(
            self.resume,
            {"is_valid": True, "errors": [], "warnings": []},
            local_resume_id=7,
            file_type="pdf",
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["resume_data"]["personal"]["firstName"], "Jane")
        self.assertEqual(response["resume_data"]["personal"]["email"], "jane@example.com")
        self.assertEqual(response["resume_data"]["experience"][0]["employer"], "Example Labs")
        self.assertEqual(response["resume_data"]["skills"], ["Python", "SQL", "Leadership"])
        self.assertEqual(response["raw_resume_schema_v1"], self.resume)
        self.assertEqual(response["raw_parser_output"], self.resume)


if __name__ == "__main__":
    unittest.main()
