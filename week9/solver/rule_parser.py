import re
from typing import Dict, Any, Tuple
from solver.defaults import DEFAULTS

def extract(problem_text: str, topic: str) -> Tuple[Dict[str, Any], float]:
    """
    Regex-based parameter extractor.
    Returns (params_dict, confidence).
    """
    params = DEFAULTS.get(topic, {}).copy()
    confidence = 0.0
    found_count = 0

    if topic == 'euler_rk4':
        patterns = {
            'omega': r'(?:omega|ω|ω|진동수|각진동수)[\s=:]*([0-9.]+)',
            'dt': r'(?:dt|time.?step|스텝|적분.?간격)[\s=:]*([0-9.]+)',
            'x0': r'(?:x0|x_0|초기.?변위|초기.?위치)[\s=:]*([0-9.]+)',
            'v0': r'(?:v0|v_0|초기.?속도)[\s=:]*([0-9.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, problem_text, re.IGNORECASE)
            if match:
                params[key] = float(match.group(1))
                found_count += 1
        
        # t_end with context check
        t_end_match = re.search(r'([0-9.]+)\s*(?:초|s|sec)', problem_text)
        if t_end_match:
            # Check if context implies total time
            context = problem_text.lower()
            if any(word in context for word in ['총', '시간', '동안', 'total', 'end', 'duration']):
                params['t_end'] = float(t_end_match.group(1))
                found_count += 1
        
        # t_end: also match "10초간", "10초 동안"
        t_end_match = re.search(r'([0-9.]+)\s*(?:초간|초\s*동안|s간|sec간)', problem_text)
        if t_end_match:
            params['t_end'] = float(t_end_match.group(1))
            found_count += 1

        confidence = found_count / 5.0
        if re.search(patterns['omega'], problem_text, re.IGNORECASE):
            confidence = max(confidence, 0.8)

    elif topic == 'planetary':
        planets = {
            '지구': (1.0, 0.017), 'earth': (1.0, 0.017),
            '화성': (1.524, 0.093), 'mars': (1.524, 0.093),
            '금성': (0.723, 0.007), 'venus': (0.723, 0.007),
            '목성': (5.203, 0.049), 'jupiter': (5.203, 0.049)
        }
        
        planet_found = False
        for name, (a, e) in planets.items():
            if name in problem_text.lower():
                params['semi_major_axis'] = a
                params['eccentricity'] = e
                planet_found = True
                break
        
        if not planet_found:
            a_match = re.search(r'(?:a|장반경|semi.?major)[\s=:]*([0-9.]+)\s*(?:AU|au)?', problem_text, re.IGNORECASE)
            if a_match:
                params['semi_major_axis'] = float(a_match.group(1))
                found_count += 1
                
            e_match = re.search(r'(?:e|이심률|eccentricity)[\s=:]*([0-9.]+)', problem_text, re.IGNORECASE)
            if e_match:
                params['eccentricity'] = float(e_match.group(1))
                found_count += 1
            
            confidence = found_count / 2.0
        else:
            confidence = 0.9

    elif topic == 'double_pendulum':
        patterns = {
            'theta1_deg': r'(?:θ1|theta1|θ₁|각도.?1|첫.?번째)[\s=:]*([0-9.]+)',
            'theta2_deg': r'(?:θ2|theta2|θ₂|각도.?2|두.?번째)[\s=:]*([0-9.]+)',
            'l1': r'(?:l1|L1|길이.?1)[\s=:]*([0-9.]+)',
            'l2': r'(?:l2|L2|길이.?2)[\s=:]*([0-9.]+)'
        }
        
        angles_found = 0
        for key, pattern in patterns.items():
            match = re.search(pattern, problem_text, re.IGNORECASE)
            if match:
                params[key] = float(match.group(1))
                if 'theta' in key:
                    angles_found += 1
                found_count += 1
        
        if angles_found == 2:
            confidence = 0.8
        elif angles_found == 1:
            confidence = 0.5
        else:
            confidence = found_count / 4.0

    elif topic == 'lagrangian':
        patterns = {
            'length': r'(?:L|길이|length)[\s=:]*([0-9.]+)\s*m?',
            'theta0_deg': r'(?:θ0|theta0|θ₀|초기.?각도?|θ)[\s=:]*([0-9.]+)',
            'mass': r'(?:질량|mass|m)[\s=:]*([0-9.]+)\s*kg?'
        }
        
        # Specific check for 'm' to avoid matching 'm' in numbers or units
        # The regex above (?:질량|mass|m)[\s=:]*([0-9.]+) might be risky.
        # Let's refine mass regex to be more specific if it's just 'm'
        
        m_match = re.search(r'(?:질량|mass)\s*[=:]?\s*([0-9.]+)', problem_text, re.IGNORECASE)
        if not m_match:
            # Only match 'm' if it's a separate word or followed by =/:
            m_match = re.search(r'\bm\s*[=:]\s*([0-9.]+)', problem_text)
            
        if m_match:
            params['mass'] = float(m_match.group(1))
            found_count += 1
            
        l_match = re.search(patterns['length'], problem_text, re.IGNORECASE)
        if l_match:
            params['length'] = float(l_match.group(1))
            found_count += 1
            
        t_match = re.search(patterns['theta0_deg'], problem_text, re.IGNORECASE)
        if t_match:
            params['theta0_deg'] = float(t_match.group(1))
            found_count += 1
            
        if l_match and t_match:
            confidence = 0.8
        else:
            confidence = found_count / 3.0

    return params, confidence
