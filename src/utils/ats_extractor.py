import re
import logging
from typing import Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ATS_SKILL_PATTERNS = [
    r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|scala|kotlin|swift|php|r|matlab|perl|shell|bash)\b',
    r'\b(html|css|react|angular|vue|node|django|flask|express|next\.js|nuxt\.js|jquery|bootstrap|tailwind)\b',
    r'\b(aws|azure|gcp|docker|kubernetes|jenkins|terraform|ansible|ci/cd|devops|cloud)\b',
    r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite|nosql|firebase)\b',
    r'\b(machine learning|deep learning|tensorflow|pytorch|scikit-learn|pandas|numpy|scipy|keras|jupyter|nlp|computer vision)\b',
    r'\b(git|github|gitlab|jira|confluence|vs code|eclipse|intellij|visual studio|figma|adobe)\b',
    r'\b(agile|scrum|kanban|waterfall|tdd|bdd|rest|api|microservices|oop|design patterns)\b',
    r'\b(linux|unix|windows|macos|networking|security|testing|qa|debugging|performance optimization)\b',
]

EXCLUDE_WORDS = {
    'software', 'engineer', 'developer', 'manager', 'analyst', 'designer', 'consultant',
    'architect', 'lead', 'senior', 'junior', 'intern', 'trainee', 'associate', 'principal',
    'director', 'vp', 'head', 'chief', 'email', 'phone', 'address', 'university', 'college',
    'company', 'corp', 'inc', 'llc', 'ltd', 'group', 'solutions', 'services', 'technologies',
    'developed', 'built', 'created', 'designed', 'implemented', 'led', 'managed', 'optimized'
}

SPECIFIC_SKILLS = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'node', 
                   'django', 'flask', 'git', 'linux', 'html', 'css', 'c++', 'c#', 'ruby', 
                   'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'mongodb', 'mysql', 
                   'postgresql', 'redis', 'elasticsearch', 'azure', 'gcp', 'kubernetes', 
                   'jenkins', 'terraform', 'ansible', 'agile', 'scrum', 'typescript', 
                   'angular', 'vue', 'express', 'pandas', 'numpy', 'tensorflow', 'pytorch',
                   'machine learning', 'deep learning', 'nlp', 'devops', 'ci/cd', 'cloud']

def is_education_line(line: str) -> bool:
    line_lower = line.lower()
    has_degree = any(d in line_lower for d in ['bs ', 'ba ', 'ms ', 'ma ', 'mba', 'phd', 'b.s.', 'b.a.', 'm.s.', 'm.a.', 'b.tech', 'm.tech', 'b.e.', 'm.e.'])
    if has_degree:
        return True
    has_university = any(u in line_lower for u in ['university', 'college', 'institute', 'school'])
    return has_university


def is_skill_line(line: str) -> bool:
    if is_education_line(line):
        return False
    line_lower = line.lower()
    skill_count = sum(1 for kw in SPECIFIC_SKILLS if kw in line_lower)
    return skill_count >= 2


def is_ats_format(text: str) -> bool:
    if not text:
        return False
    lines = text.split('\n')
    ats_indicators = 0
    pipe_count = sum(1 for line in lines if '|' in line and len(line.split('|')) > 2)
    if pipe_count > 0:
        ats_indicators += 1
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 50 and not any(h in line.lower() for h in ['experience:', 'education:', 'skills:', 'projects:']))
    if short_lines > len(lines) * 0.5:
        ats_indicators += 1
    has_traditional_headers = any(h in text.lower() for h in ['\nexperience:', '\nskills:', '\neducation:', '\nprojects:'])
    if not has_traditional_headers:
        ats_indicators += 1
    return ats_indicators >= 2


def extract_ats_skills(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    all_skills = set()
    lines = text.split('\n')
    known_skills = set()
    for pattern in ATS_SKILL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                known_skills.add(m[0].lower())
            else:
                known_skills.add(m.lower())
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if is_education_line(line):
            continue
        if re.search(r'@\w+\.\w+', line):
            continue
        if re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', line):
            continue
        if any(kw in line.lower() for kw in ['engineer', 'developer', 'manager', 'analyst', 'consultant']):
            if '|' in line or ',' not in line:
                continue
        if '|' in line and len(line.split('|')) >= 3:
            parts = [s.strip() for s in line.split('|')]
            for part in parts:
                part_lower = part.lower()
                if len(part) >= 2 and part_lower not in EXCLUDE_WORDS:
                    if part_lower in known_skills or any(p in part_lower for p in SPECIFIC_SKILLS):
                        all_skills.add(part)
        elif ',' in line and len(line.split(',')) >= 3:
            if not re.search(r'20\d{2}|19\d{2}', line):
                parts = [s.strip() for s in line.split(',')]
                for part in parts:
                    part_lower = part.lower()
                    if len(part) >= 2 and part_lower not in EXCLUDE_WORDS:
                        if part_lower in known_skills or any(p in part_lower for p in SPECIFIC_SKILLS):
                            all_skills.add(part)
    for skill in known_skills:
        if skill not in EXCLUDE_WORDS and len(skill) >= 2:
            all_skills.add(skill)
    cleaned_skills = []
    seen = set()
    for skill in all_skills:
        skill = skill.strip()
        skill_lower = skill.lower()
        if len(skill) < 2:
            continue
        if skill_lower in seen:
            continue
        if re.match(r'^\d+$', skill):
            continue
        if skill_lower in EXCLUDE_WORDS:
            continue
        cleaned_skills.append(skill)
        seen.add(skill_lower)
    if cleaned_skills:
        confidence = min(0.3 + (len(cleaned_skills) * 0.05), 1.0)
        return ', '.join(sorted(cleaned_skills)), round(confidence, 2)
    return "", 0.0



SECTION_HEADINGS = [
    'education', 'academic', 'qualification', 'degree',
    'experience', 'work', 'employment', 'professional',
    'skills', 'technologies', 'competencies',
    'projects', 'portfolio',
    'certifications', 'certificates', 'licenses',
    'awards', 'achievements', 'honors',
    'publications', 'research',
    'volunteer', 'community',
    'references', 'contact'
]

def is_section_heading(line: str) -> bool:
    line_lower = line.lower().strip()
    
    
    line_clean = re.sub(r'^[\s\-•:]+|[\s\-•:]+$', '', line_lower)
    
    
    for heading in SECTION_HEADINGS:
        if line_clean == heading or line_clean.startswith(heading + ':'):
            return True
    
    
    if line.isupper() and len(line_clean) < 30:
        return True
    
    return False


def extract_ats_education(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    education_entries = []
    lines = text.split('\n')
    
    
    in_education_section = False
    education_start_index = -1
    for idx, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        
        if is_section_heading(line) and 'education' in line_lower:
            in_education_section = True
            education_start_index = idx
            continue
        
        
        if in_education_section and is_section_heading(line):
            
            break
        
        
        if in_education_section:
            line = line.strip()
            if not line:
                continue
            
            
            if line.startswith('-') or line.startswith('•'):
                continue
            
            
            if any(kw in line.lower() for kw in ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'architect']):
                continue
            
            
            if is_skill_line(line) and not is_education_line(line):
                continue
            
            
            degree_start_pattern = re.compile(r'^(bs|ba|msc|ms|ma|mba|phd|b\.s\.|b\.a\.|m\.s\.|m\.a\.|b\.tech|m\.tech|b\.e\.|m\.e\.?)\s*(?:in\s+)?([a-zA-Z\s]+)?', re.IGNORECASE)
            year_pattern = re.compile(r'\b(19[89]\d|20[0-2]\d)\b')
            
            degree_match = degree_start_pattern.match(line)
            year_matches = year_pattern.findall(line)
            
            if degree_match and year_matches:
                degree_part = degree_match.group(1).strip()
                field_part = degree_match.group(2).strip() if degree_match.group(2) else ""
                
                if field_part:
                    entry = f"{degree_part.upper()} {field_part.strip()} ({year_matches[0]})"
                else:
                    entry = f"{degree_part.upper()} ({year_matches[0]})"
                education_entries.append(entry)
    
    
    if not education_entries:
        degree_start_pattern = re.compile(r'^(bs|ba|msc|ms|ma|mba|phd|b\.s\.|b\.a\.|m\.s\.|m\.a\.|b\.tech|m\.tech|b\.e\.|m\.e\.?)\s*(?:in\s+)?([a-zA-Z\s]+)?', re.IGNORECASE)
        year_pattern = re.compile(r'\b(19[89]\d|20[0-2]\d)\b')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('-') or line.startswith('•'):
                continue
            
            if any(kw in line.lower() for kw in ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'architect']):
                continue
            
            if is_skill_line(line) and not is_education_line(line):
                continue
            
            degree_match = degree_start_pattern.match(line)
            year_matches = year_pattern.findall(line)
            
            if degree_match and year_matches:
                degree_part = degree_match.group(1).strip()
                field_part = degree_match.group(2).strip() if degree_match.group(2) else ""
                
                if field_part:
                    entry = f"{degree_part.upper()} {field_part.strip()} ({year_matches[0]})"
                else:
                    entry = f"{degree_part.upper()} ({year_matches[0]})"
                education_entries.append(entry)
    
    if education_entries:
        seen = set()
        unique_entries = []
        for entry in education_entries:
            entry_lower = entry.lower()
            if entry_lower not in seen:
                seen.add(entry_lower)
                unique_entries.append(entry)
        
        if unique_entries:
            confidence = min(0.5 + (len(unique_entries) * 0.15), 1.0)
            return '\n'.join(unique_entries), round(confidence, 2)
    
    return "", 0.0


def extract_ats_experience(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    experience_entries = []
    lines = text.split('\n')
    title_pattern = re.compile(r'\b((?:senior|junior|lead|principal|staff)?\s*(?:software|full.?stack|front.?end|back.?end|web|data|machine.?learning|devops|cloud|product|project|marketing|sales|financial|business)?\s*(?:engineer|developer|manager|analyst|designer|architect|consultant|lead|director|specialist))\b', re.IGNORECASE)
    year_pattern = re.compile(r'\b((?:19[89]|20[0-2])\d(?:\s*[-–]\s*(?:19[89]|20[0-2])\d|present)?)\b')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if is_skill_line(line) or is_education_line(line):
            continue
        if ',' in line and len(line.split(',')) >= 3:
            if not any(kw in line.lower() for kw in ['engineer', 'developer', 'manager']):
                continue
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                title_match = title_pattern.search(parts[0])
                if title_match:
                    entry = ' | '.join(parts)
                    experience_entries.append(entry)
                    continue
                year_match = year_pattern.search(parts[-1])
                if year_match and len(parts) >= 2:
                    entry = ' | '.join(parts)
                    experience_entries.append(entry)
    
    action_verbs = ['developed', 'built', 'created', 'designed', 'implemented', 'led', 'managed', 'optimized', 'improved', 'increased', 'reduced', 'automated', 'integrated']
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•'):
            action = line.lstrip('-•').strip()
            if any(verb in action.lower() for verb in action_verbs):
                experience_entries.append(action)
    
    if experience_entries:
        seen = set()
        unique_entries = []
        for entry in experience_entries:
            entry_lower = entry.lower()[:50]
            if entry_lower not in seen:
                seen.add(entry_lower)
                unique_entries.append(entry)
        confidence = min(0.4 + (len(unique_entries) * 0.1), 1.0)
        return '\n'.join(unique_entries[:10]), round(confidence, 2)
    return "", 0.0


def extract_ats_name(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    lines = text.split('\n')
    first_line = lines[0].strip() if lines else ""
    
    if '|' in first_line:
        parts = first_line.split('|')
        if parts:
            potential_name = parts[0].strip()
            words = potential_name.split()
            if 2 <= len(words) <= 4:
                all_capitalized = all(w[0].isupper() for w in words if w)
                if all_capitalized:
                    return potential_name, 0.85
    
    name_pattern = re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$')
    match = name_pattern.match(first_line)
    if match:
        name = match.group(1)
        non_name_words = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'intern', 'specialist']
        if not any(word in name.lower() for word in non_name_words):
            return name, 0.85
    
    for line in lines[:3]:
        words = line.strip().split()
        if 2 <= len(words) <= 4:
            all_capitalized = all(w[0].isupper() for w in words if w)
            if all_capitalized:
                return ' '.join(words), 0.7
    return "", 0.0


def extract_ats_summary(text: str) -> Tuple[str, float]:
    return "", 0.0


def extract_all_ats_sections(text: str) -> Dict[str, Tuple[str, float]]:
    result = {}
    result['name'] = extract_ats_name(text)
    result['summary'] = extract_ats_summary(text)
    result['skills'] = extract_ats_skills(text)
    result['education'] = extract_ats_education(text)
    result['experience'] = extract_ats_experience(text)
    logger.info(f"ATS extraction completed: {', '.join(f'{k}:{v[1]:.2f}' for k, v in result.items())}")
    return result


def is_likely_ats_format(text: str) -> bool:
    if not text:
        return False
    indicators = 0
    if '|' in text:
        indicators += 1
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    short_no_header = sum(1 for l in lines if len(l) < 50 and not any(h in l.lower() for h in ['experience:', 'skills:', 'education:', 'projects:']))
    if short_no_header > len(lines) * 0.4:
        indicators += 1
    caps_count = sum(1 for l in lines if l.isupper() and len(l) < 30)
    if caps_count > 2:
        indicators += 1
    return indicators >= 2


if __name__ == "__main__":
    ats_resume = ""
