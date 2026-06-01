import unittest

from normalizers.builder_schema_mapper import BuilderSchemaMapper


class TestBuilderSchemaMapper(unittest.TestCase):
    def test_maps_native_output_to_builder_schema(self):
        native = {
            "name": {"raw_text": "Jane Doe", "structured_data": {"name": "Jane Doe"}, "confidence": 0.93},
            "email": {"raw_text": "[email protected]", "structured_data": {"email": "[email protected]"}, "confidence": 0.95},
            "phone": {"raw_text": "+1 555 123 4567", "structured_data": {"phone": "+1 555 123 4567"}, "confidence": 0.9},
            "summary": {"raw_text": "Backend engineer with 5 years of experience", "structured_data": {}, "confidence": 0.82},
            "skills": {
                "raw_text": "Python, Flask, SQL",
                "structured_data": {"all_skills": ["python", "flask", "sql", "Python"]},
                "confidence": 0.9,
            },
            "education": {
                "raw_text": "",
                "structured_data": {"entries": [{"institution": "ABC University", "degree": "B.Tech", "year": "2020"}]},
                "confidence": 0.86,
            },
            "experience": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {
                            "title": "Software Engineer",
                            "company": "Acme Corp",
                            "duration": "Jan 2021 - Present",
                            "description": "Built APIs",
                        }
                    ]
                },
                "confidence": 0.89,
            },
            "projects": {
                "raw_text": "",
                "structured_data": {"entries": [{"name": "Resume Parser", "description": "NLP parser", "technologies": ["Python", "Flask"]}]},
                "confidence": 0.8,
            },
            "certifications": {
                "raw_text": "",
                "structured_data": {"entries": [{"name": "AWS Certified Developer", "issuer": "AWS", "year": "2024"}]},
                "confidence": 0.83,
            },
            "languages": {
                "raw_text": "",
                "structured_data": {"languages": ["English - Fluent", "Hindi - Native"]},
                "confidence": 0.9,
            },
        }

        mapper = BuilderSchemaMapper()
        response = mapper.map_parser_output(native, overall_confidence=86.5, local_resume_id=9, file_type="pdf")

        self.assertTrue(response["success"])
        self.assertIn("resume_data", response)
        self.assertEqual(response["resume_data"]["personal"]["firstName"], "Jane")
        self.assertEqual(response["resume_data"]["personal"]["lastName"], "Doe")
        self.assertEqual(response["resume_data"]["_meta"]["local_resume_id"], 9)
        self.assertEqual(response["resume_data"]["file_type"], "pdf")
        self.assertEqual(response["resume_data"]["experience"][0]["current"], True)
        self.assertEqual(response["resume_data"]["experience"][0]["startMonth"], "January")
        self.assertEqual(response["resume_data"]["experience"][0]["startYear"], "2021")
        self.assertEqual(response["resume_data"]["languages"][0]["level"], "Fluent")
        self.assertEqual(response["resume_data"]["languages"][1]["level"], "Native")
        self.assertEqual(response["resume_data"]["skills"], ["Python", "Flask", "SQL"])

    def test_rejects_invalid_personal_and_year_fragments(self):
        native = {
            "name": {"raw_text": "john", "structured_data": {}, "confidence": 0.9},
            "email": {"raw_text": "not-an-email", "structured_data": {}, "confidence": 0.9},
            "phone": {"raw_text": "82.5%", "structured_data": {}, "confidence": 0.9},
            "contact": {"raw_text": "B.Tech Computer Science, University", "structured_data": {}, "confidence": 0.9},
            "education": {
                "raw_text": "B.Tech | Computer Science | ABC University | 82% | 20",
                "structured_data": {"entries": [{"degree": "B.Tech", "institution": "ABC University", "year": "20"}]},
                "confidence": 0.9,
            },
            "skills": {"raw_text": "", "structured_data": {"all_skills": []}, "confidence": 0.9},
            "experience": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.9},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=80, local_resume_id=1, file_type="pdf")

        personal = response["resume_data"]["personal"]
        self.assertEqual(personal["email"], "")
        self.assertEqual(personal["phone"], "")
        self.assertEqual(personal["city"], "")
        self.assertEqual(personal["country"], "")
        self.assertEqual(response["resume_data"]["education"][0]["gradYear"], "")

    def test_keeps_existing_summary_without_regeneration(self):
        native = {
            "name": {"raw_text": "Jane Doe", "structured_data": {}, "confidence": 0.9},
            "summary": {"raw_text": "Experienced payroll specialist with SAP and SQL exposure.", "structured_data": {}, "confidence": 0.2},
            "skills": {"raw_text": "", "structured_data": {"all_skills": ["sap", "sql"]}, "confidence": 0.9},
            "education": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "experience": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.0},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=75, local_resume_id=1, file_type="pdf")
        self.assertEqual(
            response["resume_data"]["summary"],
            "Experienced payroll specialist with SAP and SQL exposure.",
        )

    def test_swaps_reversed_title_employer_and_filters_language_skill(self):
        native = {
            "name": {"raw_text": "Sam Test", "structured_data": {}, "confidence": 0.9},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "skills": {
                "raw_text": "English, SAP, Pivot), Additional Information",
                "structured_data": {"all_skills": ["English", "SAP", "Pivot)", "Additional Information"]},
                "confidence": 0.9,
            },
            "education": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "experience": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {
                            "title": "ADP Solutions",
                            "company": "Analyst (UK Payroll)",
                            "duration": "Jan 2022 - Present",
                            "description": "Managed payroll operations",
                        }
                    ]
                },
                "confidence": 0.9,
            },
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "English - Fluent", "structured_data": {"languages": ["English - Fluent"]}, "confidence": 0.9},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=85, local_resume_id=1, file_type="pdf")
        exp = response["resume_data"]["experience"][0]
        self.assertEqual(exp["title"], "Analyst (UK Payroll)")
        self.assertEqual(exp["employer"], "ADP Solutions")
        self.assertEqual(exp["current"], True)
        self.assertEqual(exp["endYear"], "")
        self.assertEqual(exp["endMonth"], "")
        self.assertNotIn("English", response["resume_data"]["skills"])

    def test_fixes_neha_style_education_and_experience_shapes(self):
        native = {
            "name": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "skills": {
                "raw_text": "Requirements Gathering & Data Analysis, Payroll Auditing & Reconciliation",
                "structured_data": {"all_skills": ["Requirements Gathering & Data Analysis", "Payroll Auditing & Reconciliation"]},
                "confidence": 0.9,
            },
            "education": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {"degree": "BE in Electronics and Communications, Matrusri Engineering College (2013-2017) - 59%", "year": "2017"},
                        {"degree": "Intermediate, Narayana Junior College (2013) - 78%", "year": "2013"},
                    ]
                },
                "confidence": 0.9,
            },
            "experience": {
                "raw_text": "Analyst / Implementation Specialist @ ADP Pvt. Ltd.\n2021 - Present\nManaged end-to-end payroll implementation.",
                "structured_data": {
                    "entries": [
                        {
                            "title": "Analyst / Implementation Specialist",
                            "company": "",
                            "duration": "2021 - Present",
                            "description": "ADP Pvt. Ltd.",
                        }
                    ]
                },
                "confidence": 0.9,
            },
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.0},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=82, local_resume_id=22, file_type="docx")
        education = response["resume_data"]["education"]
        self.assertEqual(education[0]["degree"], "BE")
        self.assertEqual(education[0]["field"], "Electronics and Communications")
        self.assertEqual(education[0]["school"], "Matrusri Engineering College")
        self.assertEqual(education[0]["gpa"], "59%")
        self.assertEqual(education[0]["gradYear"], "2017")

        experience = response["resume_data"]["experience"]
        self.assertEqual(experience[0]["title"], "Analyst / Implementation Specialist")
        self.assertEqual(experience[0]["employer"], "ADP Pvt. Ltd.")
        self.assertEqual(experience[0]["startYear"], "2021")
        self.assertEqual(experience[0]["current"], True)

    def test_recovers_summary_from_raw_text_when_summary_section_missing(self):
        native = {
            "name": {"raw_text": "Alex Doe", "structured_data": {}, "confidence": 0.9},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "raw_text": {
                "raw_text": "SUMMARY\nPayroll specialist with 6 years handling HMRC and FPS/EPS.\nEXPERIENCE\nAnalyst @ ADP",
                "structured_data": {},
                "confidence": 0.9,
            },
            "skills": {"raw_text": "", "structured_data": {"all_skills": []}, "confidence": 0.9},
            "education": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "experience": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.0},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=80, local_resume_id=1, file_type="pdf")
        self.assertEqual(response["resume_data"]["summary"], "Payroll specialist with 6 years handling HMRC and FPS/EPS.")

    def test_filters_date_like_title_and_employer_and_dedupes_education_by_best_year(self):
        native = {
            "name": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "skills": {
                "raw_text": "Languages: English, Tools: SAP, SQL",
                "structured_data": {"all_skills": ["Languages: English", "Tools: SAP", "SQL"]},
                "confidence": 0.9,
            },
            "education": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {"degree": "BE in Electronics and Communications, Matrusri Engineering College (2013-2017) - 59%", "year": "2017"},
                        {"degree": "BE in Electronics and Communications, Matrusri Engineering College (2013-2017) - 59%", "year": "2013"},
                    ]
                },
                "confidence": 0.9,
            },
            "experience": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {"title": "2026 (Present)", "company": "2021", "duration": "", "description": "Managed payroll"},
                    ]
                },
                "confidence": 0.9,
            },
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": ["English - Fluent"]}, "confidence": 0.9},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=80, local_resume_id=1, file_type="pdf")
        exp = response["resume_data"]["experience"][0]
        self.assertEqual(exp["title"], "")
        self.assertEqual(exp["employer"], "")
        self.assertEqual(exp["startYear"], "2021")
        self.assertTrue(exp["current"])
        self.assertIn("SAP", response["resume_data"]["skills"])
        self.assertNotIn("English", response["resume_data"]["skills"])
        self.assertEqual(len(response["resume_data"]["education"]), 1)
        self.assertEqual(response["resume_data"]["education"][0]["gradYear"], "2017")

    def test_language_level_defaults_to_intermediate_when_missing(self):
        native = {
            "name": {"raw_text": "A B", "structured_data": {}, "confidence": 0.9},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "skills": {"raw_text": "", "structured_data": {"all_skills": []}, "confidence": 0.9},
            "education": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "experience": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "English, Hindi", "structured_data": {"languages": ["English", "Hindi"]}, "confidence": 0.9},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=90, local_resume_id=2, file_type="pdf")
        self.assertEqual(response["resume_data"]["languages"][0]["level"], "Intermediate")

    def test_merges_partial_duplicate_education_rows(self):
        native = {
            "name": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "summary": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "skills": {"raw_text": "", "structured_data": {"all_skills": []}, "confidence": 0.9},
            "education": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {"degree": "Bachelor of Commerce (B.Com)"},
                        {"school": "Indira Gandhi National Open University (IGNOU)"},
                        {"degree": "Bachelor of Commerce (B.Com)", "school": "Indira Gandhi National Open University (IGNOU)", "year": "2023"},
                        {"school": "Institute of Chartered Accountants of India"},
                        {"field": "Delhi", "school": "Institute of Chartered Accountants of India"},
                    ]
                },
                "confidence": 0.9,
            },
            "experience": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.9},
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.0},
            "email": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "phone": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
        }
        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=80, local_resume_id=28, file_type="pdf")
        edu = response["resume_data"]["education"]
        self.assertLessEqual(len(edu), 3)
        self.assertTrue(any(e.get("school") == "Indira Gandhi National Open University (IGNOU)" and e.get("gradYear") == "2023" for e in edu))

    def test_pradeepthi_style_contamination_is_filtered(self):
        cert_blob = (
            "Full Stack Web Development Udemy ServiceNow Virtual Internship Program SmartBridge & AICTE June 2025 "
            "Artificial Intelligence Certification IBM Mar 2025 PRADEEPTHI KOPPISETTY Full Stack Web Developer "
            "pradeepthikoppisetty05@gmail.com PROFESSIONAL SUMMARY Results-driven developer"
        )
        native = {
            "name": {"raw_text": "Languages Python Javascript Php", "structured_data": {}, "confidence": 0.9},
            "raw_text": {"raw_text": cert_blob, "structured_data": {}, "confidence": 0.9},
            "summary": {"raw_text": "Results-driven Full Stack Web Developer.", "structured_data": {}, "confidence": 0.9},
            "email": {"raw_text": "pradeepthikoppisetty05@gmail.com", "structured_data": {}, "confidence": 0.9},
            "phone": {"raw_text": "9177313626", "structured_data": {}, "confidence": 0.9},
            "skills": {
                "raw_text": "Tools & Platforms: Git, Databases: MySQL, HTML5, CSS3, VS, Code",
                "structured_data": {"all_skills": ["Tools & Platforms: Git", "Databases: MySQL", "HTML5", "CSS3", "VS", "Code"]},
                "confidence": 0.9,
            },
            "education": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {
                            "degree": "Intermediate",
                            "field": "Bachelors of Technology in Computer",
                            "institution": "Siddhartha Institute of Technology and Sciences, Telangana",
                            "gpa": "GPA: 8.5",
                        }
                    ]
                },
                "confidence": 0.9,
            },
            "experience": {
                "raw_text": "",
                "structured_data": {
                    "entries": [
                        {
                            "title": "Web Developer Intern",
                            "company": "",
                            "description": "Hungry Bird IT Consultancy and Services, Hyderabad, Telangana Developed and maintained dynamic web applications using React.js.",
                        }
                    ]
                },
                "confidence": 0.9,
            },
            "projects": {"raw_text": "", "structured_data": {"entries": []}, "confidence": 0.0},
            "certifications": {
                "raw_text": "",
                "structured_data": {"entries": []},
                "confidence": 0.9,
            },
            "achievements": {"raw_text": "", "structured_data": {}, "confidence": 0.0},
            "languages": {"raw_text": "", "structured_data": {"languages": []}, "confidence": 0.0},
        }

        response = BuilderSchemaMapper().map_parser_output(native, overall_confidence=90, local_resume_id=33, file_type="pdf")
        data = response["resume_data"]
        self.assertEqual(data["personal"]["firstName"], "Pradeepthi")
        self.assertEqual(data["personal"]["lastName"], "Koppisetty")
        self.assertEqual(data["experience"][0]["employer"], "Hungry Bird IT Consultancy and Services")
        self.assertEqual(data["education"][0]["degree"], "Bachelors of Technology")
        self.assertEqual(data["education"][0]["field"], "Computer")
        self.assertNotIn("Languages", data["personal"]["firstName"])
        self.assertTrue(any(c["name"] == "Artificial Intelligence Certification" for c in data["certifications"]))
        self.assertFalse(any("PROFESSIONAL SUMMARY" in c["name"] for c in data["certifications"]))
        self.assertIn("Git", data["skills"])
        self.assertNotIn("Vs", data["skills"])


if __name__ == "__main__":
    unittest.main()
