import os
import json
import anthropic
from typing import Dict, Any

# Default parameters for each topic
DEFAULTS = {
    'euler_rk4': {'omega': 2.0, 'dt': 0.05, 't_end': 10.0, 'x0': 1.0, 'v0': 0.0},
    'planetary': {'semi_major_axis': 1.0, 'eccentricity': 0.2, 'mass_star': 1.989e30},
    'double_pendulum': {
        'theta1_deg': 120.0, 'theta2_deg': -30.0, 
        'l1': 1.0, 'l2': 1.0, 'm1': 1.0, 'm2': 1.0, 
        't_end': 20.0, 'compare_chaos': True
    },
    'lagrangian': {'length': 1.0, 'mass': 1.0, 'theta0_deg': 60.0, 'omega0': 0.0, 't_end': 10.0}
}

SYSTEM_PROMPT = r"""
You are a highly precise physics parameter extraction engine. Your goal is to analyze natural language descriptions of classical mechanics problems and extract numerical parameters required for computational solvers.

### Topic Schemas:

1. **euler_rk4** (Simple Harmonic Oscillator / ODE Solver):
   - 'omega': Angular frequency (rad/s). Default: 2.0.
   - 'dt': Time step for integration (s). Default: 0.05.
   - 't_end': Total simulation time (s). Default: 10.0.
   - 'x0': Initial position (m). Default: 1.0.
   - 'v0': Initial velocity (m/s). Default: 0.0.

2. **planetary** (Planetary Motion / Keplerian Orbit):
   - 'semi_major_axis': Semi-major axis of the orbit (AU). Default: 1.0.
   - 'eccentricity': Orbital eccentricity (dimensionless, 0 to 1). Default: 0.2.
   - 'mass_star': Mass of the central star (kg). Default: 1.989e30 (1 Solar Mass).

3. **double_pendulum** (Chaotic Double Pendulum):
   - 'theta1_deg': Initial angle of the first pendulum (degrees). Default: 120.0.
   - 'theta2_deg': Initial angle of the second pendulum (degrees). Default: -30.0.
   - 'l1': Length of the first rod (m). Default: 1.0.
   - 'l2': Length of the second rod (m). Default: 1.0.
   - 'm1': Mass of the first bob (kg). Default: 1.0.
   - 'm2': Mass of the second bob (kg). Default: 1.0.
   - 't_end': Total simulation time (s). Default: 20.0.
   - 'compare_chaos': Whether to run a second simulation with a tiny perturbation (bool). Default: True.

4. **lagrangian** (Single Pendulum via Lagrangian Mechanics):
   - 'length': Length of the pendulum (m). Default: 1.0.
   - 'mass': Mass of the bob (kg). Default: 1.0.
   - 'theta0_deg': Initial angle (degrees). Default: 60.0.
   - 'omega0': Initial angular velocity (rad/s). Default: 0.0.
   - 't_end': Total simulation time (s). Default: 10.0.

### Instructions:
- Identify the numerical values mentioned in the problem description.
- Convert all units to standard SI units (meters, kilograms, seconds) or the specific units requested in the schema (e.g., degrees for angles, AU for semi-major axis).
- If a parameter is not mentioned, do not include it in the JSON; it will be filled by defaults.
- Return ONLY a valid JSON object. No preamble, no explanation.

### Physics Context for Advanced Extraction:
Classical mechanics is governed by Newton's laws and the principle of least action. 
- For the Simple Harmonic Oscillator (euler_rk4), the equation of motion is $\ddot{x} + \omega^2 x = 0$. Numerical stability depends heavily on the time step 'dt' relative to the period $T = 2\pi/\omega$.
- Planetary motion (planetary) follows Kepler's laws. The semi-major axis 'a' and eccentricity 'e' define the elliptical path: $r(\theta) = a(1-e^2)/(1+e\cos\theta)$.
- The double pendulum (double_pendulum) is a classic example of deterministic chaos. Small changes in initial conditions ($\theta_1, \theta_2$) lead to exponentially diverging trajectories, characterized by Lyapunov exponents.
- Lagrangian mechanics (lagrangian) uses $L = T - V$ to derive equations of motion. For a pendulum, $L = \frac{1}{2}ml^2\dot{\theta}^2 - mgl(1-\cos\theta)$.

When users describe problems, they might use conversational language. "A 5kg mass on a 2 meter string released from 45 degrees" implies mass=5.0, length=2.0, theta0_deg=45.0. "Run it for a minute" implies t_end=60.0. "A highly elliptical orbit around a sun-like star" implies high eccentricity (e.g., 0.8) and mass_star=1.989e30.

Precision is paramount. If a user says "half a second step", set dt=0.5. If they say "milli-second precision", set dt=0.001.

Ensure the JSON is strictly formatted.
"""

def parse(problem_text: str, topic: str) -> Dict[str, Any]:
    """
    Extracts numerical parameters from natural language physics problem text using Claude.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    defaults = DEFAULTS.get(topic, {})
    
    if not api_key:
        return defaults

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nProblem: {problem_text}\nReturn JSON with extracted parameters only."
                }
            ]
        )
        
        extracted_text = response.content[0].text.strip()
        # Find JSON boundaries if Claude added markdown
        if "```json" in extracted_text:
            extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
        elif "{" in extracted_text:
            extracted_text = extracted_text[extracted_text.find("{"):extracted_text.rfind("}")+1]
            
        params = json.loads(extracted_text)
        
        # Merge with defaults
        result = defaults.copy()
        result.update(params)
        return result
        
    except Exception:
        # On ANY error, return defaults silently
        return defaults
