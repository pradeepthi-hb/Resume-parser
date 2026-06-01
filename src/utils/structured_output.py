
import re
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SectionData:
    raw_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    method: str = "regex"
    error: Optional[str] = None


@dataclass
class ResumeData:
    name: SectionData = field(default_factory=SectionData)
    email: SectionData = field(default_factory=SectionData)
    phone: SectionData = field(default_factory=SectionData)
    summary: SectionData = field(default_factory=SectionData)
    skills: SectionData = field(default_factory=SectionData)
    education: SectionData = field(default_factory=SectionData)
    experience: SectionData = field(default_factory=SectionData)
    projects: SectionData = field(default_factory=SectionData)
    certifications: SectionData = field(default_factory=SectionData)
    languages: SectionData = field(default_factory=SectionData)
    interests: SectionData = field(default_factory=SectionData)
    achievements: SectionData = field(default_factory=SectionData)
    publications: SectionData = field(default_factory=SectionData)
    volunteer: SectionData = field(default_factory=SectionData)
    raw_text: SectionData = field(default_factory=SectionData)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, SectionData):
                result[key] = {
                    "raw_text": value.raw_text,
                    "structured_data": value.structured_data,
                    "confidence": value.confidence,
                    "method": value.method,
                    "error": value.error
                }
            else:
                result[key] = value
        return result
    
    def get_confidence_summary(self) -> Dict[str, float]:
        return {
            key: value.confidence 
            for key, value in asdict(self).items() 
            if isinstance(value, SectionData)
        }
    
    def get_average_confidence(self) -> float:
        scores = [
            value.confidence 
            for key, value in asdict(self).items() 
            if isinstance(value, SectionData) and value.confidence > 0
        ]
        return sum(scores) / len(scores) if scores else 0.0


class StructuredOutputGenerator:
    
    
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    
    PHONE_PATTERNS = [
        re.compile(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
        re.compile(r'\+?[0-9]{1,4}[-.\s]?[0-9]{2,4}[-.\s]?[0-9]{2,4}[-.\s]?[0-9]{2,4}'),
        re.compile(r'\b[0-9]{10,12}\b'),
    ]
    
    
    LINKEDIN_PATTERN = re.compile(r'linkedin\.com/in/[A-Za-z0-9_-]+', re.IGNORECASE)
    
    
    GITHUB_PATTERN = re.compile(r'github\.com/[A-Za-z0-9_-]+', re.IGNORECASE)
    
    
    WEBSITE_PATTERN = re.compile(r'https?://[^\s]+', re.IGNORECASE)

    @staticmethod
    def _is_noise_company(candidate: str) -> bool:
        if not candidate:
            return True
        lowered = candidate.lower()
        if len(candidate.split()) > 8:
            return True
        noise_terms = {
            "responsible",
            "developed",
            "managed",
            "implemented",
            "budgeting",
            "forecasting",
            "process",
            "processes",
            "duties",
            "achievements",
            "projects",
        }
        return any(term in lowered for term in noise_terms)
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_email(self, text: str) -> Tuple[str, float]:
        match = self.EMAIL_PATTERN.search(text)
        if match:
            return match.group(0), 0.95
        return "", 0.0
    
    def extract_phone(self, text: str) -> Tuple[str, float]:
        for pattern in self.PHONE_PATTERNS:
            match = pattern.search(text)
            if match:
                phone = match.group(0)
                
                phone = re.sub(r'[^\d+]', '', phone)
                if len(phone) >= 10:
                    return phone, 0.90
        return "", 0.0
    
    def extract_links(self, text: str) -> Dict[str, List[str]]:
        links = {
            "linkedin": [],
            "github": [],
            "website": []
        }
        
        
        linkedin_matches = self.LINKEDIN_PATTERN.findall(text)
        links["linkedin"] = list(set(linkedin_matches))
        
        
        github_matches = self.GITHUB_PATTERN.findall(text)
        links["github"] = list(set(github_matches))
        
        
        website_matches = self.WEBSITE_PATTERN.findall(text)
        for site in website_matches:
            if 'linkedin' not in site.lower() and 'github' not in site.lower():
                links["website"].append(site)
        
        return links
    
    def parse_skills(self, skills_text: str) -> Dict[str, Any]:
        if not skills_text:
            return {"categories": {}, "all_skills": []}
        
        lines = skills_text.split('\n')
        all_skills = []
        categories = {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "tools": [],
            "cloud": [],
            "soft_skills": [],
            "other": []
        }
        
        
        category_patterns = {
            "programming_languages": ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin', 'scala', 'typescript', 'matlab'],
            "frameworks": ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node', 'express', 'rails', 'laravel', '.net', 'asp.net'],
            "databases": ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite', 'cassandra', 'dynamodb'],
            "tools": ['git', 'docker', 'kubernetes', 'jenkins', 'jira', 'terraform', 'ansible', 'maven', 'gradle', 'npm', 'yarn'],
            "cloud": ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'heroku', 'digitalocean'],
            "soft_skills": ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'management']
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            
            skills = re.split(r'[,;â€¢\n]+', line)
            for skill in skills:
                skill = skill.strip().lower()
                if not skill or len(skill) < 2:
                    continue
                
                all_skills.append(skill)
                
                
                categorized = False
                for category, keywords in category_patterns.items():
                    if any(re.search(rf'\b{re.escape(kw)}\b', skill) for kw in keywords):
                        categories[category].append(skill)
                        categorized = True
                        break
                
                if not categorized:
                    categories["other"].append(skill)
        
        return {
            "categories": {k: list(set(v)) for k, v in categories.items()},
            "all_skills": list(set(all_skills)),
            "total_count": len(set(all_skills))
        }
    
    def parse_education(self, education_text: str) -> List[Dict[str, Any]]:
        if not education_text:
            return []
        
        entries = []
        lines = education_text.split('\n')
        
        current_entry = {}
        current_field = ""
        
        
        degree_patterns = [
            (r'\bphd\b|\bdoctorate\b', 'phd'),
            (r'\bmaster\b|\bm\.sc\b|\bmba\b|\bm\.tech\b|\bm\.e\.', 'masters'),
            (r'\bbachelor\b|\bb\.sc\b|\bb\.tech\b|\bb\.e\.|\bba\b|\bbba\b', 'bachelors'),
            (r'\bdiploma\b', 'diploma'),
            (r'\bcertificate\b', 'certificate')
        ]
        
        
        year_pattern = re.compile(r'(?:19|20)\d{2}')
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                continue
            
            
            for pattern, degree_level in degree_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {"degree": line, "level": degree_level}
                    current_field = "degree"
                    break
            else:
                
                years = year_pattern.findall(line)
                if years and 'year' not in current_entry:
                    current_entry["year"] = ' - '.join(years) if len(years) > 1 else years[0]
                
                
                institution_keywords = ['university', 'college', 'institute', 'school', 'academy']
                if any(kw in line.lower() for kw in institution_keywords):
                    current_entry["institution"] = line
                
                
                if 'gpa' in line.lower() or 'grade' in line.lower() or '%' in line:
                    current_entry["grade"] = line
        
        if current_entry:
            entries.append(current_entry)
        
        return entries

    def parse_experience(self, experience_text: str) -> List[Dict[str, Any]]:
        if not experience_text:
            return []

        entries = []
        lines = experience_text.split('\n')
        current_entry = {}

        company_patterns = [
            r'\b(?:company|employer|organization)\s*[:\-]\s*([A-Z][A-Za-z0-9&,\.\-\s]{1,60})$',
            r'@\s*([A-Z][A-Za-z0-9&,\.\-\s]{1,60})$',
            r'\bat\s+([A-Z][A-Za-z0-9&,\.\-\s]{1,60})(?:\s*(?:\||\-|–|from|to|present|current|$))',
        ]

        date_pattern = re.compile(
            r'(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(?:19|20)\d{2}'
            r'(?:\s*[-–to]+\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?'
            r'(?:19|20)\d{2}|present|current)?',
            re.IGNORECASE
        )

        title_keywords = [
            'engineer', 'developer', 'manager', 'analyst', 'designer',
            'consultant', 'architect', 'lead', 'director', 'head',
            'chief', 'specialist', 'executive', 'accountant'
        ]

        company_suffixes = (
            'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation',
            'company', 'technologies', 'solutions', 'services', 'pvt',
            'bank', 'group', 'consulting'
        )
        action_terms = (
            'responsible', 'developed', 'managed', 'implemented', 'worked',
            'budgeting', 'forecasting', 'designed', 'led', 'handled'
        )

        def is_company_like_line(candidate: str) -> bool:
            cleaned = candidate.strip()
            if not cleaned or len(cleaned) > 60:
                return False
            if re.search(r'\d', cleaned):
                return False
            lowered = cleaned.lower()
            if any(term in lowered for term in action_terms):
                return False
            words = cleaned.split()
            if len(words) > 6:
                return False
            if any(lowered.endswith(sfx) or f" {sfx}" in lowered for sfx in company_suffixes):
                return True
            titled_words = [w for w in words if w and w[0].isalpha()]
            if titled_words and all(w[0].isupper() for w in titled_words):
                return True
            return False

        def is_probable_entry_start(candidate: str, existing_entry: Dict[str, Any]) -> bool:
            if not existing_entry:
                return False
            lower_candidate = candidate.lower()
            has_existing_core = any(k in existing_entry for k in ("title", "company", "duration"))
            if not has_existing_core:
                return False

            has_title_line = any(keyword in lower_candidate for keyword in title_keywords) and len(candidate) <= 80
            has_date_line = bool(date_pattern.search(candidate))
            has_company_header = bool(
                re.search(r'\b(?:company|employer|organization)\s*[:\-]', candidate, re.IGNORECASE)
                or re.search(r'@[A-Z]', candidate)
            )

            if has_date_line:
                return "duration" in existing_entry and candidate != existing_entry.get("duration", "")
            if has_title_line:
                return (
                    "title" in existing_entry
                    and candidate != existing_entry.get("title", "")
                    and ("duration" in existing_entry or "description" in existing_entry or "company" in existing_entry)
                )
            if has_company_header:
                return "company" in existing_entry and candidate != existing_entry.get("company", "")
            if is_company_like_line(candidate):
                return "company" in existing_entry and ("duration" in existing_entry or "description" in existing_entry)
            return False

        for line in lines:
            line = line.strip()
            if not line:
                if current_entry and len(current_entry) > 1:
                    entries.append(current_entry)
                    current_entry = {}
                continue

            if is_probable_entry_start(line, current_entry):
                entries.append(current_entry)
                current_entry = {}

            line_lower = line.lower()
            consumed = False
            for keyword in title_keywords:
                if keyword in line_lower and len(line) < 70:
                    current_entry["title"] = line
                    consumed = True
                    break

            if date_pattern.search(line):
                current_entry["duration"] = line
                consumed = True

            for pattern in company_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    company_candidate = match.group(1).strip(" -|,")
                    if not self._is_noise_company(company_candidate):
                        current_entry["company"] = company_candidate
                        consumed = True
                        break

            if (
                "company" not in current_entry
                and is_company_like_line(line)
                and not any(keyword in line_lower for keyword in title_keywords)
            ):
                current_entry["company"] = line
                consumed = True

            if not consumed:
                if "description" not in current_entry:
                    current_entry["description"] = []
                current_entry["description"].append(line)

        if current_entry:
            entries.append(current_entry)

        for entry in entries:
            if "description" in entry and isinstance(entry["description"], list):
                entry["description"] = " ".join(entry["description"])

        return entries

    def parse_projects(self, projects_text: str) -> List[Dict[str, Any]]:
        if not projects_text:
            return []

        entries = []
        lines = projects_text.split('\n')

        current_project = {}

        tech_pattern = re.compile(
            r'\b(python|java|javascript|react|angular|vue|node|django|flask|mongodb|mysql|postgresql|aws|azure|gcp|docker|kubernetes)\b',
            re.IGNORECASE,
        )

        link_pattern = re.compile(r'(github\.com|gitlab\.com|heroku\.com|aws\.amazon\.com|bitbucket\.org)[^\s]*', re.IGNORECASE)

        for line in lines:
            line = line.strip()
            if not line:
                if current_project and len(current_project) > 0:
                    entries.append(current_project)
                    current_project = {}
                continue

            if len(line) < 60 and not line.startswith('-'):
                current_project["name"] = line
            else:
                techs = tech_pattern.findall(line)
                if techs:
                    if "technologies" not in current_project:
                        current_project["technologies"] = list(set([t.lower() for t in techs]))

                links = link_pattern.findall(line)
                if links:
                    current_project["link"] = links[0]

                if "description" not in current_project:
                    current_project["description"] = []
                current_project["description"].append(line)

        if current_project:
            entries.append(current_project)

        for entry in entries:
            if "description" in entry and isinstance(entry["description"], list):
                entry["description"] = " ".join(entry["description"])

        return entries
    
    def parse_certifications(self, cert_text: str) -> List[Dict[str, Any]]:
        if not cert_text:
            return []
        
        entries = []
        lines = cert_text.split('\n')
        
        
        cert_patterns = [
            r'\b(aws|azure|gcp|google|microsoft|amazon)\b.*\b(certified|certification|certificate)\b',
            r'\b(pmp|scrum|agile|pmi|itil|ccna|ccnp|mcse|mcsa|ceh|cissp|cisa|comptia)\b',
            r'\b(certified|certificate)\b.*\b(developer|engineer|architect|manager|professional)\b'
        ]
        
        year_pattern = re.compile(r'(?:19|20)\d{2}')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            entry = {"name": line}
            
            
            years = year_pattern.findall(line)
            if years:
                entry["year"] = years[0]
            
            
            issuers = ['aws', 'azure', 'google', 'microsoft', 'oracle', 'salesforce', 'scrum', 'pmi']
            for issuer in issuers:
                if issuer in line.lower():
                    entry["issuer"] = issuer.title()
            
            entries.append(entry)
        
        return entries
    
    def generate_structured_output(
        self,
        raw_text: str,
        name: str,
        skills: str,
        education: str,
        experience: str,
        projects: str,
        certifications: str,
        languages: str = "",
        interests: str = "",
        achievements: str = "",
        publications: str = "",
        volunteer: str = "",
        name_confidence: float = 0.0,
        skills_confidence: float = 0.0,
        education_confidence: float = 0.0,
        experience_confidence: float = 0.0,
        projects_confidence: float = 0.0,
        certifications_confidence: float = 0.0,
        languages_confidence: float = 0.0,
        interests_confidence: float = 0.0,
        achievements_confidence: float = 0.0,
        publications_confidence: float = 0.0,
        volunteer_confidence: float = 0.0,
        text_confidence: float = 0.0,
        extraction_method: str = "hybrid"
    ) -> ResumeData:
        
        resume = ResumeData()
        
        
        email, email_conf = self.extract_email(raw_text)
        phone, phone_conf = self.extract_phone(raw_text)
        links = self.extract_links(raw_text)
        
        
        resume.raw_text = SectionData(
            raw_text=raw_text,
            structured_data={"word_count": len(raw_text.split())},
            confidence=text_confidence,
            method=extraction_method
        )
        
        
        resume.name = SectionData(
            raw_text=name,
            structured_data={"name": name},
            confidence=name_confidence,
            method="nlp" if name_confidence > 0.5 else "regex"
        )
        
        
        resume.email = SectionData(
            raw_text=email,
            structured_data={"email": email},
            confidence=email_conf,
            method="regex"
        )
        
        
        resume.phone = SectionData(
            raw_text=phone,
            structured_data={"phone": phone},
            confidence=phone_conf,
            method="regex"
        )
        
        
        structured_skills = self.parse_skills(skills)
        resume.skills = SectionData(
            raw_text=skills,
            structured_data=structured_skills,
            confidence=skills_confidence,
            method="hybrid"
        )
        
        
        structured_education = self.parse_education(education)
        resume.education = SectionData(
            raw_text=education,
            structured_data={"entries": structured_education},
            confidence=education_confidence,
            method="hybrid"
        )
        
        
        structured_experience = self.parse_experience(experience)
        resume.experience = SectionData(
            raw_text=experience,
            structured_data={"entries": structured_experience},
            confidence=experience_confidence,
            method="hybrid"
        )
        
        
        structured_projects = self.parse_projects(projects)
        resume.projects = SectionData(
            raw_text=projects,
            structured_data={"entries": structured_projects},
            confidence=projects_confidence,
            method="hybrid"
        )
        
        
        structured_certs = self.parse_certifications(certifications)
        resume.certifications = SectionData(
            raw_text=certifications,
            structured_data={"entries": structured_certs},
            confidence=certifications_confidence,
            method="hybrid"
        )
        
        
        resume.languages = SectionData(
            raw_text=languages,
            structured_data={"languages": [l.strip() for l in languages.split('\n') if l.strip()]},
            confidence=languages_confidence,
            method="regex"
        )
        
        
        resume.interests = SectionData(
            raw_text=interests,
            structured_data={"interests": [i.strip() for i in interests.split('\n') if i.strip()]},
            confidence=interests_confidence,
            method="regex"
        )
        
        
        resume.achievements = SectionData(
            raw_text=achievements,
            structured_data={"achievements": [a.strip() for a in achievements.split('\n') if a.strip()]},
            confidence=achievements_confidence,
            method="regex"
        )
        
        
        resume.publications = SectionData(
            raw_text=publications,
            structured_data={"publications": [p.strip() for p in publications.split('\n') if p.strip()]},
            confidence=publications_confidence,
            method="regex"
        )
        
        
        resume.volunteer = SectionData(
            raw_text=volunteer,
            structured_data={"experience": [v.strip() for v in volunteer.split('\n') if v.strip()]},
            confidence=volunteer_confidence,
            method="regex"
        )
        
        
        if links["linkedin"] or links["github"] or links["website"]:
            resume.summary = SectionData(
                raw_text=resume.summary.raw_text,
                structured_data={
                    **resume.summary.structured_data,
                    "links": links
                },
                confidence=resume.summary.confidence,
                method=resume.summary.method
            )
        
        return resume



_structured_output_generator = None

def get_structured_output_generator() -> StructuredOutputGenerator:
    global _structured_output_generator
    if _structured_output_generator is None:
        _structured_output_generator = StructuredOutputGenerator()
    return _structured_output_generator


def generate_structured_resume(
    raw_text: str,
    extracted_sections: Dict[str, Tuple[str, float]]
) -> Dict[str, Any]:
    generator = get_structured_output_generator()
    
    
    name = extracted_sections.get("name", ("", 0.0))
    skills = extracted_sections.get("skills", ("", 0.0))
    education = extracted_sections.get("education", ("", 0.0))
    experience = extracted_sections.get("experience", ("", 0.0))
    projects = extracted_sections.get("projects", ("", 0.0))
    certifications = extracted_sections.get("certifications", ("", 0.0))
    languages = extracted_sections.get("languages", ("", 0.0))
    interests = extracted_sections.get("interests", ("", 0.0))
    achievements = extracted_sections.get("achievements", ("", 0.0))
    publications = extracted_sections.get("publications", ("", 0.0))
    volunteer = extracted_sections.get("volunteer", ("", 0.0))
    
    
    resume = generator.generate_structured_output(
        raw_text=raw_text,
        name=name[0],
        skills=skills[0],
        education=education[0],
        experience=experience[0],
        projects=projects[0],
        certifications=certifications[0],
        languages=languages[0],
        interests=interests[0],
        achievements=achievements[0],
        publications=publications[0],
        volunteer=volunteer[0],
        name_confidence=name[1],
        skills_confidence=skills[1],
        education_confidence=education[1],
        experience_confidence=experience[1],
        projects_confidence=projects[1],
        certifications_confidence=certifications[1],
        languages_confidence=languages[1],
        interests_confidence=interests[1],
        achievements_confidence=achievements[1],
        publications_confidence=publications[1],
        volunteer_confidence=volunteer[1],
        text_confidence=extracted_sections.get("text", (0.0, 0.0))[1]
    )
    
    return resume.to_dict()

if __name__ == "__main__":
    pass
