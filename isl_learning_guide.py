"""
ISL Learning Guide & Real-Time Posture Coach
Provides comprehensive finger posture specifications, visual diagrams,
and real-time anatomical evaluation with actionable improvement coaching
for Indian Sign Language (ISL) letters, digits, and gestures.
"""

import numpy as np
import cv2

# Comprehensive ISL Learning Guide Database
ISL_SYMBOL_GUIDES = {
    # -------------------------------------------------------------
    # VOWELS (Two-handed: Right Index touches Left Fingertips)
    # -------------------------------------------------------------
    'A': {
        'name': 'Letter A (ISL)',
        'category': 'Alphabet (Vowel)',
        'two_handed': True,
        'dominant_hand': 'Right Hand (Pointer)',
        'non_dominant_hand': 'Left Hand (Base)',
        'summary': 'Right index finger touches the tip of the Left Thumb.',
        'target_touch': 'Left Thumb Tip',
        'steps': [
            '1. Open your non-dominant (left) hand facing the camera with fingers spread.',
            '2. Make a loose fist with your dominant (right) hand, extending only your index finger.',
            '3. Touch your right index fingertip directly to the tip of your left thumb.'
        ],
        'left_hand_req': {
            'thumb': 'Extended outward / upward',
            'index': 'Open / Spread',
            'middle': 'Open / Spread',
            'ring': 'Open / Spread',
            'pinky': 'Open / Spread',
            'palm': 'Facing front / slightly inward'
        },
        'right_hand_req': {
            'thumb': 'Folded against fingers',
            'index': 'Straight & Pointing at Left Thumb',
            'middle': 'Curled into palm',
            'ring': 'Curled into palm',
            'pinky': 'Curled into palm',
            'palm': 'Facing toward left hand'
        },
        'tips': [
            'Ensure your left hand is clearly visible and not tilted sideways.',
            'Keep your right middle, ring, and pinky fingers tightly folded so the system clearly detects only the index pointer.',
            'Maintain steady contact between the fingertips for at least 1.5 seconds.'
        ],
        'touch_finger': 'thumb'
    },
    'E': {
        'name': 'Letter E (ISL)',
        'category': 'Alphabet (Vowel)',
        'two_handed': True,
        'dominant_hand': 'Right Hand (Pointer)',
        'non_dominant_hand': 'Left Hand (Base)',
        'summary': 'Right index finger touches the tip of the Left Index finger.',
        'target_touch': 'Left Index Fingertip',
        'steps': [
            '1. Open your left hand facing forward with fingers spread apart.',
            '2. Point your right index finger with other fingers folded.',
            '3. Touch your right index fingertip directly to the tip of your left index finger.'
        ],
        'left_hand_req': {
            'thumb': 'Extended outward',
            'index': 'Extended upward (Target finger)',
            'middle': 'Open / Spread',
            'ring': 'Open / Spread',
            'pinky': 'Open / Spread',
            'palm': 'Facing front'
        },
        'right_hand_req': {
            'thumb': 'Curled',
            'index': 'Extended & Touching Left Index tip',
            'middle': 'Curled',
            'ring': 'Curled',
            'pinky': 'Curled',
            'palm': 'Facing left'
        },
        'tips': [
            'Avoid touching the middle finger by mistake; target strictly the left index tip.'
        ],
        'touch_finger': 'index'
    },
    'I': {
        'name': 'Letter I (ISL)',
        'category': 'Alphabet (Vowel)',
        'two_handed': True,
        'dominant_hand': 'Right Hand (Pointer)',
        'non_dominant_hand': 'Left Hand (Base)',
        'summary': 'Right index finger touches the tip of the Left Middle finger.',
        'target_touch': 'Left Middle Fingertip',
        'steps': [
            '1. Spread all fingers of your left hand open facing forward.',
            '2. Point your right index finger forward with other fingers curled.',
            '3. Touch your right index fingertip directly to the tip of your left middle (tallest) finger.'
        ],
        'left_hand_req': {
            'thumb': 'Extended',
            'index': 'Extended',
            'middle': 'Extended (Target finger)',
            'ring': 'Extended',
            'pinky': 'Extended',
            'palm': 'Facing front'
        },
        'right_hand_req': {
            'thumb': 'Curled',
            'index': 'Pointing at Left Middle tip',
            'middle': 'Curled',
            'ring': 'Curled',
            'pinky': 'Curled',
            'palm': 'Facing left'
        },
        'tips': ['Aim for the very tip of the middle finger.'],
        'touch_finger': 'middle'
    },
    'O': {
        'name': 'Letter O (ISL)',
        'category': 'Alphabet (Vowel)',
        'two_handed': True,
        'dominant_hand': 'Right Hand (Pointer)',
        'non_dominant_hand': 'Left Hand (Base)',
        'summary': 'Right index finger touches the tip of the Left Ring finger.',
        'target_touch': 'Left Ring Fingertip',
        'steps': [
            '1. Open your left hand with fingers spread comfortably.',
            '2. Point your right index finger.',
            '3. Touch your right index fingertip to the tip of your left ring finger.'
        ],
        'left_hand_req': {
            'thumb': 'Extended',
            'index': 'Extended',
            'middle': 'Extended',
            'ring': 'Extended (Target finger)',
            'pinky': 'Extended',
            'palm': 'Facing front'
        },
        'right_hand_req': {
            'thumb': 'Curled',
            'index': 'Pointing at Left Ring tip',
            'middle': 'Curled',
            'ring': 'Curled',
            'pinky': 'Curled',
            'palm': 'Facing left'
        },
        'tips': ['Keep the left pinky separated so the right index clearly isolates the ring finger.'],
        'touch_finger': 'ring'
    },
    'U': {
        'name': 'Letter U (ISL)',
        'category': 'Alphabet (Vowel)',
        'two_handed': True,
        'dominant_hand': 'Right Hand (Pointer)',
        'non_dominant_hand': 'Left Hand (Base)',
        'summary': 'Right index finger touches the tip of the Left Pinky finger.',
        'target_touch': 'Left Pinky Fingertip',
        'steps': [
            '1. Open your left hand with fingers spread.',
            '2. Point your right index finger.',
            '3. Touch your right index fingertip to the tip of your left pinky finger.'
        ],
        'left_hand_req': {
            'thumb': 'Extended',
            'index': 'Extended',
            'middle': 'Extended',
            'ring': 'Extended',
            'pinky': 'Extended (Target finger)',
            'palm': 'Facing front'
        },
        'right_hand_req': {
            'thumb': 'Curled',
            'index': 'Pointing at Left Pinky tip',
            'middle': 'Curled',
            'ring': 'Curled',
            'pinky': 'Curled',
            'palm': 'Facing left'
        },
        'tips': ['Touch the outermost small finger (pinky) on your left hand.'],
        'touch_finger': 'pinky'
    },

    # -------------------------------------------------------------
    # CONSONANTS (Two-handed / Single-handed)
    # -------------------------------------------------------------
    'B': {
        'name': 'Letter B (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Both hands form circular loops with thumb & index, touching together like glasses or "8".',
        'target_touch': 'Thumb-to-Thumb & Index-to-Index',
        'steps': [
            '1. On your left hand, touch the tip of your thumb to the tip of your index finger to make a circle (curl remaining 3 fingers).',
            '2. On your right hand, do the same: touch thumb to index to make another circle.',
            '3. Bring both hands together so both circular loops touch side-by-side or tip-to-tip.'
        ],
        'left_hand_req': {'thumb': 'Touching left index', 'index': 'Touching left thumb (loop)', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing right'},
        'right_hand_req': {'thumb': 'Touching right index', 'index': 'Touching right thumb (loop)', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing left'},
        'tips': ['Form two distinct circular eyelets and press the index and thumb knuckles together.'],
        'touch_finger': 'loop'
    },
    'C': {
        'name': 'Letter C (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right hand forms a curved "C" shape resting against flat vertical left palm.',
        'target_touch': 'Right Hand against Left Palm',
        'steps': [
            '1. Hold your left hand flat, palm facing right.',
            '2. Curve your right hand into a "C" cup shape (fingers together, thumb curved opposite).',
            '3. Place the open side of your right "C" curved hand against your flat left palm.'
        ],
        'left_hand_req': {'thumb': 'Flat / alongside', 'index': 'Flat', 'middle': 'Flat', 'ring': 'Flat', 'pinky': 'Flat', 'palm': 'Facing right'},
        'right_hand_req': {'thumb': 'Curved', 'index': 'Curved forward', 'middle': 'Curved forward', 'ring': 'Curved forward', 'pinky': 'Curved forward', 'palm': 'Facing left'},
        'tips': ['Keep the right hand curve clear like grasping a cup.'],
        'touch_finger': 'palm'
    },
    'D': {
        'name': 'Letter D (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right vertical index finger placed against Left hand circle loop.',
        'target_touch': 'Right Index against Left Thumb-Index Loop',
        'steps': [
            '1. Make a circle with your left thumb and left index finger.',
            '2. Point your right index finger straight up vertically.',
            '3. Rest your right vertical index finger against the right edge of the left hand loop.'
        ],
        'left_hand_req': {'thumb': 'Touching left index (loop)', 'index': 'Curved in loop', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing right'},
        'right_hand_req': {'thumb': 'Curled', 'index': 'Straight Vertical Up', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing forward'},
        'tips': ['The right vertical index forms the stem of the lowercase "d" while the left hand forms the rounded belly.'],
        'touch_finger': 'loop'
    },
    'L': {
        'name': 'Letter L (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right hand forms an "L" shape (thumb & index at 90°) placed on flat left palm.',
        'target_touch': 'Right "L" on Left Palm',
        'steps': [
            '1. Extend your right thumb and right index finger at a 90-degree right angle to form an "L".',
            '2. Keep other three right fingers curled into the palm.',
            '3. Hold your left hand flat, palm facing forward or up, and rest your right "L" against it.'
        ],
        'left_hand_req': {'thumb': 'Flat', 'index': 'Flat', 'middle': 'Flat', 'ring': 'Flat', 'pinky': 'Flat', 'palm': 'Facing front / upward'},
        'right_hand_req': {'thumb': 'Extended 90° sideways', 'index': 'Extended straight up', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing forward'},
        'tips': ['Keep the angle between thumb and index as close to 90 degrees as possible.'],
        'touch_finger': 'palm'
    },
    'M': {
        'name': 'Letter M (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Three right fingers (index, middle, ring) laid across open left palm.',
        'target_touch': '3 Right Fingers on Left Palm',
        'steps': [
            '1. Hold your left hand flat and horizontal, palm facing upward.',
            '2. On your right hand, extend 3 fingers together (index, middle, ring) and curl thumb and pinky.',
            '3. Lay the three right fingertips flat across the center of your open left palm.'
        ],
        'left_hand_req': {'thumb': 'Open', 'index': 'Open', 'middle': 'Open', 'ring': 'Open', 'pinky': 'Open', 'palm': 'Facing upward'},
        'right_hand_req': {'thumb': 'Curled under', 'index': 'Extended downward', 'middle': 'Extended downward', 'ring': 'Extended downward', 'pinky': 'Curled', 'palm': 'Facing left palm'},
        'tips': ['Think of 3 legs of the letter "m" resting on the palm.'],
        'touch_finger': 'palm'
    },
    'N': {
        'name': 'Letter N (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Two right fingers (index, middle) laid across open left palm.',
        'target_touch': '2 Right Fingers on Left Palm',
        'steps': [
            '1. Hold your left hand flat and horizontal, palm facing upward.',
            '2. On your right hand, extend only 2 fingers (index and middle) and curl thumb, ring, and pinky.',
            '3. Lay the two right fingertips flat across the center of your open left palm.'
        ],
        'left_hand_req': {'thumb': 'Open', 'index': 'Open', 'middle': 'Open', 'ring': 'Open', 'pinky': 'Open', 'palm': 'Facing upward'},
        'right_hand_req': {'thumb': 'Curled', 'index': 'Extended downward', 'middle': 'Extended downward', 'ring': 'Curled into palm', 'pinky': 'Curled into palm', 'palm': 'Facing left palm'},
        'tips': ['Think of 2 legs of the letter "n" resting on the palm.'],
        'touch_finger': 'palm'
    },
    'V': {
        'name': 'Letter V (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right hand peace sign "V" (index & middle spread) resting on flat left palm.',
        'target_touch': 'Right "V" on Left Palm',
        'steps': [
            '1. Hold your left hand flat, palm facing upward.',
            '2. Form a "V" (peace sign) with your right hand: extend index and middle fingers spread apart.',
            '3. Place the tips of your right "V" fingers onto your flat left palm.'
        ],
        'left_hand_req': {'thumb': 'Open', 'index': 'Open', 'middle': 'Open', 'ring': 'Open', 'pinky': 'Open', 'palm': 'Facing upward'},
        'right_hand_req': {'thumb': 'Folded over ring finger', 'index': 'Extended & spread', 'middle': 'Extended & spread', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing left/down'},
        'tips': ['Spread index and middle fingers wide into a distinct "V" shape.'],
        'touch_finger': 'palm'
    },
    'W': {
        'name': 'Letter W (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right hand three fingers (index, middle, ring spread) resting on flat left palm.',
        'target_touch': 'Right "W" on Left Palm',
        'steps': [
            '1. Hold your left hand flat, palm facing upward.',
            '2. On your right hand, spread index, middle, and ring fingers apart forming a "W".',
            '3. Place the tips of the three right fingers onto your flat left palm.'
        ],
        'left_hand_req': {'thumb': 'Open', 'index': 'Open', 'middle': 'Open', 'ring': 'Open', 'pinky': 'Open', 'palm': 'Facing upward'},
        'right_hand_req': {'thumb': 'Folded over pinky', 'index': 'Extended & spread', 'middle': 'Extended & spread', 'ring': 'Extended & spread', 'pinky': 'Curled', 'palm': 'Facing left/down'},
        'tips': ['Keep the three fingers evenly spaced to clearly form a "W".'],
        'touch_finger': 'palm'
    },
    'X': {
        'name': 'Letter X (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Both index fingers extended and crossed over each other forming an "X".',
        'target_touch': 'Crossed Index Fingers',
        'steps': [
            '1. Extend your left index finger, curling all other left fingers.',
            '2. Extend your right index finger, curling all other right fingers.',
            '3. Cross the right index finger over the left index finger at a 90-degree angle to make an "X".'
        ],
        'left_hand_req': {'thumb': 'Curled', 'index': 'Extended straight', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing body'},
        'right_hand_req': {'thumb': 'Curled', 'index': 'Extended straight across', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing body'},
        'tips': ['Keep other fingers firmly tucked so only the two crossed index fingers are visible.'],
        'touch_finger': 'index'
    },
    'Y': {
        'name': 'Letter Y (ISL)',
        'category': 'Alphabet (Consonant)',
        'two_handed': True,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Left Hand',
        'summary': 'Right index finger placed directly into the thumb-index webbing of the Left hand.',
        'target_touch': 'Webbing between Left Thumb and Left Index',
        'steps': [
            '1. Hold your left hand flat facing forward with your left thumb spread wide apart.',
            '2. Point your right index finger forward with other right fingers curled.',
            '3. Place your right index fingertip into the V-shaped webbing between the left thumb and index finger.'
        ],
        'left_hand_req': {'thumb': 'Spread wide at 60-90°', 'index': 'Straight up', 'middle': 'Up', 'ring': 'Up', 'pinky': 'Up', 'palm': 'Facing forward'},
        'right_hand_req': {'thumb': 'Curled', 'index': 'Pointing directly into thumb web', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing left'},
        'tips': ['The left thumb and index form a "V", and the right index forms the stem of "Y".'],
        'touch_finger': 'web'
    },

    # -------------------------------------------------------------
    # DIGITS (0 to 5)
    # -------------------------------------------------------------
    '0': {
        'name': 'Number 0',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Closed fist with all 5 fingers folded into palm.',
        'target_touch': 'None (Closed Fist)',
        'steps': [
            '1. Hold your dominant hand in front of camera.',
            '2. Curl all four fingers tightly into your palm.',
            '3. Fold your thumb across your fingers forming a firm zero fist.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Curled across fingers', 'index': 'Curled into palm', 'middle': 'Curled into palm', 'ring': 'Curled into palm', 'pinky': 'Curled into palm', 'palm': 'Facing camera'},
        'tips': ['Ensure no fingertips are sticking out.'],
        'touch_finger': None
    },
    '1': {
        'name': 'Number 1',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Index finger pointed straight up, other 3 fingers and thumb curled.',
        'target_touch': 'None (Index Up)',
        'steps': [
            '1. Point your index finger straight up toward the ceiling.',
            '2. Fold your thumb over your middle, ring, and pinky fingers.',
            '3. Keep palm facing forward toward the camera.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Folded over curled fingers', 'index': 'Straight up vertically', 'middle': 'Curled', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing camera'},
        'tips': ['Hold index finger tall and vertical.'],
        'touch_finger': None
    },
    '2': {
        'name': 'Number 2',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Index and middle fingers straight up (V shape), other fingers curled.',
        'target_touch': 'None (Index + Middle Up)',
        'steps': [
            '1. Extend your index and middle fingers straight up.',
            '2. Keep your ring and pinky fingers curled into your palm.',
            '3. Fold your thumb over your ring finger.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Folded', 'index': 'Straight up', 'middle': 'Straight up', 'ring': 'Curled', 'pinky': 'Curled', 'palm': 'Facing camera'},
        'tips': ['Keep both fingers parallel or slightly spread.'],
        'touch_finger': None
    },
    '3': {
        'name': 'Number 3',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Thumb, index, and middle fingers extended; ring and pinky curled.',
        'target_touch': 'None (Thumb + Index + Middle)',
        'steps': [
            '1. Extend your thumb outward to the side.',
            '2. Extend your index and middle fingers straight up.',
            '3. Keep ring and pinky curled into your palm.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Extended sideways', 'index': 'Extended up', 'middle': 'Extended up', 'ring': 'Curled into palm', 'pinky': 'Curled into palm', 'palm': 'Facing camera'},
        'tips': ['Make sure your thumb is distinctly spread outward.'],
        'touch_finger': None
    },
    '4': {
        'name': 'Number 4',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Four fingers straight up (index, middle, ring, pinky); thumb folded.',
        'target_touch': 'None (4 Fingers Up)',
        'steps': [
            '1. Extend index, middle, ring, and pinky fingers straight up.',
            '2. Fold your thumb flat across your palm.',
            '3. Face your palm toward the camera.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Folded across palm', 'index': 'Straight up', 'middle': 'Straight up', 'ring': 'Straight up', 'pinky': 'Straight up', 'palm': 'Facing camera'},
        'tips': ['Keep the four fingers parallel and thumb tucked in.'],
        'touch_finger': None
    },
    '5': {
        'name': 'Number 5',
        'category': 'Digits (0-9)',
        'two_handed': False,
        'dominant_hand': 'Dominant Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'All 5 fingers open and spread wide facing camera.',
        'target_touch': 'None (Open Palm)',
        'steps': [
            '1. Open your dominant hand completely.',
            '2. Spread all five fingers (thumb, index, middle, ring, pinky) wide apart.',
            '3. Hold palm steady facing directly at the camera.'
        ],
        'left_hand_req': {'thumb': 'N/A', 'index': 'N/A', 'middle': 'N/A', 'ring': 'N/A', 'pinky': 'N/A', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Open & spread', 'index': 'Open & spread', 'middle': 'Open & spread', 'ring': 'Open & spread', 'pinky': 'Open & spread', 'palm': 'Facing camera'},
        'tips': ['Spread fingers comfortably apart so all 5 are distinct.'],
        'touch_finger': None
    },

    # -------------------------------------------------------------
    # DYNAMIC SIGNS (Common Phrases)
    # -------------------------------------------------------------
    'hello': {
        'name': 'Hello / Greeting',
        'category': 'Dynamic Signs',
        'two_handed': False,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Open hand moves outward from temple / forehead in a friendly wave/salute.',
        'target_touch': 'Temple / Forehead outward',
        'steps': [
            '1. Start with open flat right hand near your right temple / forehead.',
            '2. Palm faces slightly forward and outward.',
            '3. Move hand smoothly outward and forward in a salute or wave gesture.'
        ],
        'left_hand_req': {'thumb': 'Resting', 'index': 'Resting', 'middle': 'Resting', 'ring': 'Resting', 'pinky': 'Resting', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Open flat', 'index': 'Open flat', 'middle': 'Open flat', 'ring': 'Open flat', 'pinky': 'Open flat', 'palm': 'Near temple moving outward'},
        'tips': ['Perform a smooth, confident motion away from the head.'],
        'touch_finger': None
    },
    'thank you': {
        'name': 'Thank You',
        'category': 'Dynamic Signs',
        'two_handed': False,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Fingertips touch chin/lips, then move forward toward the conversational partner.',
        'target_touch': 'Chin to outward forward',
        'steps': [
            '1. Touch the fingertips of your flat right hand to your chin or lower lip.',
            '2. Move your hand gently forward and slightly downward toward the viewer.',
            '3. Finish with open palm facing upward/forward.'
        ],
        'left_hand_req': {'thumb': 'Resting', 'index': 'Resting', 'middle': 'Resting', 'ring': 'Resting', 'pinky': 'Resting', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Flat alongside', 'index': 'Fingertips to chin', 'middle': 'Fingertips to chin', 'ring': 'Fingertips to chin', 'pinky': 'Flat', 'palm': 'Facing body then outward'},
        'tips': ['Start cleanly at the chin and extend outward with gratitude.'],
        'touch_finger': None
    },
    'alright': {
        'name': 'Alright / Okay',
        'category': 'Dynamic Signs',
        'two_handed': False,
        'dominant_hand': 'Right Hand',
        'non_dominant_hand': 'Not required',
        'summary': 'Dominant hand forms a clear thumbs-up gesture.',
        'target_touch': 'None (Thumbs Up)',
        'steps': [
            '1. Curl all four fingers tightly into your palm.',
            '2. Point your thumb straight up.',
            '3. Hold your hand in center view with confidence.'
        ],
        'left_hand_req': {'thumb': 'Resting', 'index': 'Resting', 'middle': 'Resting', 'ring': 'Resting', 'pinky': 'Resting', 'palm': 'Resting'},
        'right_hand_req': {'thumb': 'Pointing straight up', 'index': 'Curled tightly', 'middle': 'Curled tightly', 'ring': 'Curled tightly', 'pinky': 'Curled tightly', 'palm': 'Facing sideways'},
        'tips': ['Ensure your thumb points directly upward.'],
        'touch_finger': None
    }
}


def get_symbol_categories():
    """Return dictionary of categories with their symbol list."""
    cats = {
        'Vowels (A, E, I, O, U)': ['A', 'E', 'I', 'O', 'U'],
        'Consonants': ['B', 'C', 'D', 'L', 'M', 'N', 'V', 'W', 'X', 'Y'],
        'Numbers (0–5)': ['0', '1', '2', '3', '4', '5'],
        'Common Signs': ['hello', 'thank you', 'alright']
    }
    return cats


def generate_hand_svg(symbol):
    """
    Generate an intuitive SVG visual graphic showing the target finger configuration,
    highlighting active fingers and contact points.
    """
    guide = ISL_SYMBOL_GUIDES.get(symbol, ISL_SYMBOL_GUIDES.get(str(symbol).upper(), ISL_SYMBOL_GUIDES['A']))
    is_two_handed = guide.get('two_handed', False)

    # Color palette
    bg_color = "#0f172a"
    hand_color = "#334155"
    active_color = "#38bdf8"
    accent_gold = "#f59e0b"
    touch_glow = "#ec4899"
    text_color = "#f8fafc"

    svg_lines = [
        f'<svg viewBox="0 0 540 260" xmlns="http://www.w3.org/2000/svg" style="background:{bg_color}; border-radius:12px; width:100%; height:auto; box-shadow:0 8px 24px rgba(0,0,0,0.3);">'
    ]

    svg_lines.append('<defs>')
    svg_lines.append('  <linearGradient id="gradAct" x1="0%" y1="0%" x2="100%" y2="100%">')
    svg_lines.append(f'    <stop offset="0%" stop-color="{active_color}" />')
    svg_lines.append(f'    <stop offset="100%" stop-color="{accent_gold}" />')
    svg_lines.append('  </linearGradient>')
    svg_lines.append('  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">')
    svg_lines.append('    <feGaussianBlur stdDeviation="5" result="blur" />')
    svg_lines.append('    <feComposite in="SourceGraphic" in2="blur" operator="over" />')
    svg_lines.append('  </filter>')
    svg_lines.append('</defs>')

    # Title & Target Banner
    svg_lines.append(f'<text x="270" y="28" fill="{text_color}" font-family="sans-serif" font-size="16" font-weight="700" text-anchor="middle">TARGET POSTURE: {guide["name"].upper()}</text>')
    svg_lines.append(f'<text x="270" y="48" fill="#94a3b8" font-family="sans-serif" font-size="12" text-anchor="middle">{guide["summary"]}</text>')

    if is_two_handed:
        # LEFT HAND (Base) on Left Side (center around X=150, Y=145)
        svg_lines.append(f'<rect x="40" y="65" width="210" height="175" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1.5" />')
        svg_lines.append(f'<text x="145" y="86" fill="{active_color}" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">LEFT HAND (Base)</text>')

        # Draw Left Palm Outline
        svg_lines.append(f'<path d="M 120 180 Q 120 145 130 135 L 130 115 Q 130 105 140 105 Q 150 105 150 115 L 150 135 L 160 135 L 160 100 Q 160 90 170 90 Q 180 90 180 100 L 180 140 L 190 145 L 190 110 Q 190 102 198 102 Q 206 102 206 110 L 206 148 L 214 152 L 214 125 Q 214 118 221 118 Q 228 118 228 125 L 228 180 Q 220 215 170 215 Q 125 215 120 180 Z" fill="{hand_color}" stroke="#475569" stroke-width="2" />')

        # Left Thumb
        thumb_highlight = (guide.get('touch_finger') == 'thumb')
        t_fill = active_color if thumb_highlight else hand_color
        t_stroke = accent_gold if thumb_highlight else "#475569"
        svg_lines.append(f'<path d="M 122 165 Q 90 150 85 132 Q 82 120 94 115 Q 106 115 118 135 Z" fill="{t_fill}" stroke="{t_stroke}" stroke-width="2" />')
        if thumb_highlight:
            svg_lines.append(f'<circle cx="88" cy="123" r="10" fill="{touch_glow}" opacity="0.4" filter="url(#glow)" />')
            svg_lines.append(f'<circle cx="88" cy="123" r="5" fill="{accent_gold}" />')
            svg_lines.append(f'<text x="88" y="105" fill="{accent_gold}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">TOUCH TIP</text>')

        # Highlight other finger tips if target
        tf = guide.get('touch_finger')
        coords = {'index': (140, 105), 'middle': (170, 90), 'ring': (198, 102), 'pinky': (221, 118), 'palm': (165, 165), 'web': (110, 140)}
        if tf in coords and tf != 'thumb':
            cx, cy = coords[tf]
            svg_lines.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{touch_glow}" opacity="0.5" filter="url(#glow)" />')
            svg_lines.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{accent_gold}" />')
            svg_lines.append(f'<text x="{cx}" y="{cy-12}" fill="{accent_gold}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">TOUCH HERE</text>')

        # RIGHT HAND (Pointer) on Right Side (center around X=390, Y=145)
        svg_lines.append(f'<rect x="290" y="65" width="210" height="175" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1.5" />')
        svg_lines.append(f'<text x="395" y="86" fill="{accent_gold}" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">RIGHT HAND (Pointer)</text>')

        # Right Hand: Index pointing leftwards toward target
        svg_lines.append(f'<path d="M 440 180 Q 450 145 420 135 L 340 130 Q 325 130 325 140 Q 325 150 340 150 L 415 155 Q 425 180 435 205 Q 450 205 440 180 Z" fill="{active_color}" stroke="{accent_gold}" stroke-width="2" />')
        # Curled fingers
        svg_lines.append(f'<ellipse cx="420" cy="165" rx="18" ry="12" fill="{hand_color}" stroke="#475569" stroke-width="1.5" />')
        svg_lines.append(f'<ellipse cx="415" cy="180" rx="18" ry="12" fill="{hand_color}" stroke="#475569" stroke-width="1.5" />')
        svg_lines.append(f'<ellipse cx="408" cy="195" rx="18" ry="12" fill="{hand_color}" stroke="#475569" stroke-width="1.5" />')

        # Pointer tip indicator
        svg_lines.append(f'<circle cx="330" cy="140" r="7" fill="{accent_gold}" filter="url(#glow)" />')
        svg_lines.append(f'<line x1="330" y1="140" x2="255" y2="135" stroke="{accent_gold}" stroke-width="2" stroke-dasharray="4,4" />')
        svg_lines.append(f'<text x="270" y="125" fill="{accent_gold}" font-family="sans-serif" font-size="10" font-weight="bold" text-anchor="middle">POINTS TO</text>')

    else:
        # SINGLE HAND DIAGRAM (Center)
        svg_lines.append(f'<rect x="150" y="65" width="240" height="175" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1.5" />')
        svg_lines.append(f'<text x="270" y="86" fill="{active_color}" font-family="sans-serif" font-size="13" font-weight="600" text-anchor="middle">{guide["dominant_hand"].upper()}</text>')

        # Render Single Hand depending on symbol
        if symbol == '0':
            # Fist
            svg_lines.append(f'<circle cx="270" cy="155" r="40" fill="{active_color}" stroke="{accent_gold}" stroke-width="3" />')
            svg_lines.append(f'<text x="270" y="160" fill="{bg_color}" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">CLOSED FIST</text>')
        elif symbol == '1':
            # Index up
            svg_lines.append(f'<rect x="260" y="95" width="20" height="50" rx="10" fill="{active_color}" stroke="{accent_gold}" stroke-width="2" />')
            svg_lines.append(f'<ellipse cx="270" cy="170" rx="35" ry="30" fill="{hand_color}" stroke="#475569" stroke-width="2" />')
            svg_lines.append(f'<text x="270" y="90" fill="{accent_gold}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">INDEX UP</text>')
        elif symbol == 'alright':
            # Thumbs Up
            svg_lines.append(f'<rect x="260" y="95" width="22" height="45" rx="11" fill="{accent_gold}" stroke="#ffffff" stroke-width="2" />')
            svg_lines.append(f'<ellipse cx="270" cy="165" rx="36" ry="28" fill="{active_color}" stroke="{accent_gold}" stroke-width="2" />')
            svg_lines.append(f'<text x="270" y="85" fill="{accent_gold}" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">THUMBS UP 👍</text>')
        elif symbol == 'hello':
            # Temple / Forehead Wave
            svg_lines.append(f'<circle cx="270" cy="140" r="46" fill="#0f172a" stroke="{active_color}" stroke-width="2" />')
            svg_lines.append(f'<text x="270" y="112" fill="{accent_gold}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">TEMPLE / FOREHEAD</text>')
            svg_lines.append(f'<path d="M 245 145 L 285 145 L 285 130 L 310 152 L 285 174 L 285 159 L 245 159 Z" fill="{active_color}" />')
            svg_lines.append(f'<text x="270" y="202" fill="#94a3b8" font-family="sans-serif" font-size="11" text-anchor="middle">Wave outward from head</text>')
        elif symbol == 'thank you':
            # Chin to Outward Forward
            svg_lines.append(f'<circle cx="270" cy="115" r="32" fill="#0f172a" stroke="#10b981" stroke-width="2" />')
            svg_lines.append(f'<text x="270" y="118" fill="{accent_gold}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">CHIN / LIPS</text>')
            svg_lines.append(f'<path d="M 270 152 L 270 190 L 255 190 L 270 215 L 285 190 L 270 190 Z" fill="#10b981" />')
            svg_lines.append(f'<text x="270" y="232" fill="#94a3b8" font-family="sans-serif" font-size="11" text-anchor="middle">Extend forward from chin</text>')
        else:
            # Generic Hand
            svg_lines.append(f'<ellipse cx="270" cy="155" rx="42" ry="48" fill="{active_color}" stroke="{accent_gold}" stroke-width="2" />')
            svg_lines.append(f'<text x="270" y="160" fill="{bg_color}" font-family="sans-serif" font-size="13" font-weight="bold" text-anchor="middle">{symbol.upper()}</text>')

    svg_lines.append('</svg>')
    return "".join(svg_lines)


def evaluate_hand_posture(results, target_symbol):
    """
    Evaluates detected MediaPipe hand landmarks against target symbol requirements.
    Returns:
        {
            'accuracy': float (0-100),
            'status': str ('PERFECT', 'GOOD', 'NEEDS_WORK', 'NO_HANDS'),
            'primary_hint': str (Immediate actionable tip),
            'checklist': list of dicts: [{'label': str, 'passed': bool, 'detail': str}],
            'target_touch_coords': (x, y) or None,
            'pointer_coords': (x, y) or None
        }
    """
    from fingerspelling import get_hand_metrics

    target_symbol = str(target_symbol).strip()
    guide = ISL_SYMBOL_GUIDES.get(target_symbol, ISL_SYMBOL_GUIDES.get(target_symbol.upper(), ISL_SYMBOL_GUIDES['A']))
    is_two_handed = guide.get('two_handed', False)

    lh_lms = None
    rh_lms = None

    if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) >= 21:
        lh_lms = results.left_hand_landmarks.landmark
    if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) >= 21:
        rh_lms = results.right_hand_landmarks.landmark

    if not lh_lms and not rh_lms and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            if len(hand_lms.landmark) >= 21:
                label = "Right"
                if hasattr(results, 'multi_handedness') and idx < len(results.multi_handedness):
                    label = results.multi_handedness[idx].classification[0].label
                if label == "Left" and not lh_lms:
                    lh_lms = hand_lms.landmark
                elif not rh_lms:
                    rh_lms = hand_lms.landmark

    lh = get_hand_metrics(lh_lms)
    rh = get_hand_metrics(rh_lms)

    checklist = []
    hints = []
    score_components = []
    target_touch_coords = None
    pointer_coords = None

    # Case 1: No hands detected at all
    if not lh and not rh:
        return {
            'accuracy': 0.0,
            'status': 'NO_HANDS',
            'primary_hint': '👋 Hold your hand(s) up in front of the camera.',
            'checklist': [
                {'label': 'Hand(s) Detected', 'passed': False, 'detail': 'No hands detected in camera frame.'}
            ],
            'target_touch_coords': None,
            'pointer_coords': None
        }

    # Case 2: Two-Handed Symbol Evaluation
    if is_two_handed:
        # Check 1: Both hands present
        has_both = (lh is not None and rh is not None)
        checklist.append({
            'label': 'Both Hands in View',
            'passed': has_both,
            'detail': 'Both Left and Right hands clearly visible.' if has_both else 'Please bring both hands into camera view.'
        })
        score_components.append(25.0 if has_both else (10.0 if (lh or rh) else 0.0))

        if not has_both:
            # Check if both wrists are raised in front of torso via pose landmarks
            has_both_wrists = False
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                if len(pose_lms) > 16:
                    l_w, r_w = pose_lms[15], pose_lms[16]
                    if l_w.y < 0.85 and r_w.y < 0.85:
                        has_both_wrists = True

            if has_both_wrists:
                hint = "💡 Fingertips overlapping! Separate hands slightly so camera sees both."
            else:
                missing = "Left (base) hand" if lh is None else "Right (pointer) hand"
                hint = f'⚠️ Bring your {missing} into the camera view.'
            return {
                'accuracy': 25.0,
                'status': 'NEEDS_WORK',
                'primary_hint': hint,
                'checklist': checklist,
                'target_touch_coords': None,
                'pointer_coords': None
            }

        # Hands are present
        l_span = lh["span"]
        r_span = rh["span"]
        avg_span = (l_span + r_span) / 2.0

        touch_target = guide.get('touch_finger', 'thumb')

        # Dynamically identify Pointer Hand vs Base Hand
        # A pointer hand has index extended and other fingers curled (ext_count <= 2)
        # A base hand has multiple fingers open (ext_count >= 2 or higher than pointer)
        is_rh_pointer = rh["index_ext"] and (not rh["pinky_ext"] or rh["ext_count"] <= 2)
        is_lh_pointer = lh["index_ext"] and (not lh["pinky_ext"] or lh["ext_count"] <= 2)

        if is_rh_pointer and not is_lh_pointer:
            pointer_hand, base_hand = rh, lh
            pointer_label_name = "Right Hand (Pointer)"
            base_label_name = "Left Hand (Base)"
        elif is_lh_pointer and not is_rh_pointer:
            pointer_hand, base_hand = lh, rh
            pointer_label_name = "Left Hand (Pointer)"
            base_label_name = "Right Hand (Base)"
        else:
            if rh["ext_count"] <= lh["ext_count"]:
                pointer_hand, base_hand = rh, lh
                pointer_label_name = "Right Hand (Pointer)"
                base_label_name = "Left Hand (Base)"
            else:
                pointer_hand, base_hand = lh, rh
                pointer_label_name = "Left Hand (Pointer)"
                base_label_name = "Right Hand (Base)"

        p_index_tip = pointer_hand["tips"]["index"]
        pointer_coords = (float(p_index_tip[0]), float(p_index_tip[1]))

        # Check 2: Dominant / Pointer hand posture according to target symbol
        target_sym_upper = target_symbol.upper()
        if target_sym_upper == 'V':
            d_im = np.linalg.norm(pointer_hand["tips"]["index"] - pointer_hand["tips"]["middle"]) / pointer_hand["span"]
            pointer_ok = pointer_hand["index_ext"] and pointer_hand["middle_ext"] and not pointer_hand["ring_ext"] and not pointer_hand["pinky_ext"] and (d_im >= 0.28)
            checklist.append({
                'label': f'Peace Sign "V" Formed ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Index and middle extended and spread in a V.' if pointer_ok else 'Spread index and middle fingers apart into a V shape.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if (pointer_hand["index_ext"] and pointer_hand["middle_ext"]) else 0.0))
            if not (pointer_hand["index_ext"] and pointer_hand["middle_ext"]):
                hints.append('✌️ Extend both your index and middle fingers.')
            elif d_im < 0.28:
                hints.append('✌️ Spread your index and middle fingers apart into a clear "V".')
            elif pointer_hand["ring_ext"] or pointer_hand["pinky_ext"]:
                hints.append('👉 Fold ring and pinky fingers into your palm.')

        elif target_sym_upper == 'N':
            d_im = np.linalg.norm(pointer_hand["tips"]["index"] - pointer_hand["tips"]["middle"]) / pointer_hand["span"]
            pointer_ok = pointer_hand["index_ext"] and pointer_hand["middle_ext"] and not pointer_hand["ring_ext"] and not pointer_hand["pinky_ext"] and (d_im < 0.35)
            checklist.append({
                'label': f'Two Fingers Together "N" ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Two fingers held together side-by-side.' if pointer_ok else 'Hold index and middle fingers together side-by-side.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if (pointer_hand["index_ext"] and pointer_hand["middle_ext"]) else 0.0))
            if not (pointer_hand["index_ext"] and pointer_hand["middle_ext"]):
                hints.append('✌️ Extend both your index and middle fingers together.')
            elif pointer_hand["ring_ext"] or pointer_hand["pinky_ext"]:
                hints.append('👉 Fold ring and pinky fingers into your palm.')

        elif target_sym_upper == 'W':
            d_ir = np.linalg.norm(pointer_hand["tips"]["index"] - pointer_hand["tips"]["ring"]) / pointer_hand["span"]
            pointer_ok = pointer_hand["index_ext"] and pointer_hand["middle_ext"] and pointer_hand["ring_ext"] and not pointer_hand["pinky_ext"] and (d_ir >= 0.45)
            checklist.append({
                'label': f'Three Fingers Spread "W" ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Index, middle, and ring spread apart.' if pointer_ok else 'Extend index, middle, and ring spread in a W.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if (pointer_hand["index_ext"] and pointer_hand["middle_ext"]) else 0.0))
            if not (pointer_hand["index_ext"] and pointer_hand["middle_ext"] and pointer_hand["ring_ext"]):
                hints.append('🖐️ Extend 3 fingers: index, middle, and ring.')
            elif d_ir < 0.45:
                hints.append('🖐️ Spread the three fingers apart into a distinct "W".')

        elif target_sym_upper == 'M':
            d_ir = np.linalg.norm(pointer_hand["tips"]["index"] - pointer_hand["tips"]["ring"]) / pointer_hand["span"]
            pointer_ok = pointer_hand["index_ext"] and pointer_hand["middle_ext"] and pointer_hand["ring_ext"] and not pointer_hand["pinky_ext"] and (d_ir < 0.50)
            checklist.append({
                'label': f'Three Fingers Together "M" ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Index, middle, and ring held together.' if pointer_ok else 'Keep three fingers together side-by-side.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if (pointer_hand["index_ext"] and pointer_hand["middle_ext"]) else 0.0))
            if not (pointer_hand["index_ext"] and pointer_hand["middle_ext"] and pointer_hand["ring_ext"]):
                hints.append('🖐️ Extend 3 fingers: index, middle, and ring held together.')

        elif target_sym_upper == 'L':
            pointer_ok = pointer_hand["thumb_ext"] and pointer_hand["index_ext"] and not pointer_hand["middle_ext"] and not pointer_hand["pinky_ext"]
            checklist.append({
                'label': f'"L" Shape Formed ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Thumb and index at 90-degree angle.' if pointer_ok else 'Extend thumb and index at 90 degrees.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if pointer_hand["index_ext"] else 0.0))
            if not (pointer_hand["thumb_ext"] and pointer_hand["index_ext"]):
                hints.append('👉 Form an "L" shape with thumb and index finger.')

        elif target_sym_upper == 'C':
            pointer_ok = not pointer_hand["index_ext"] and not pointer_hand["pinky_ext"]
            checklist.append({
                'label': f'Curved "C" Shape ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Fingers curved forward in a cup shape.' if pointer_ok else 'Curve your fingers into a "C" cup.'
            })
            score_components.append(25.0 if pointer_ok else 12.0)
            if not pointer_ok:
                hints.append('🤏 Curve all fingers and thumb into a "C" cup shape.')

        elif target_sym_upper == 'B':
            loop_p = np.linalg.norm(pointer_hand["tips"]["thumb"] - pointer_hand["tips"]["index"]) < avg_span * 0.50
            loop_b = np.linalg.norm(base_hand["tips"]["thumb"] - base_hand["tips"]["index"]) < avg_span * 0.50
            checklist.append({
                'label': 'Double Loops Formed (Letter B)',
                'passed': bool(loop_p and loop_b),
                'detail': 'Both hands forming circular loops.' if (loop_p and loop_b) else 'Touch thumb to index on both hands to form circles.'
            })
            score_components.append(25.0 if (loop_p and loop_b) else (15.0 if (loop_p or loop_b) else 0.0))
            if not (loop_p and loop_b):
                hints.append('👌 Make a circle with thumb and index on both hands.')

        else:
            # Standard single index pointer for vowels (A, E, I, O, U) and D, X, Y
            pointer_ok = pointer_hand["index_ext"] and not pointer_hand["middle_ext"] and not pointer_hand["pinky_ext"]
            checklist.append({
                'label': f'Index Pointer Formed ({pointer_label_name})',
                'passed': bool(pointer_ok),
                'detail': 'Index pointing forward, remaining fingers curled.' if pointer_ok else 'Extend index finger; curl remaining fingers into palm.'
            })
            score_components.append(25.0 if pointer_ok else (12.0 if pointer_hand["index_ext"] else 0.0))
            if not pointer_hand["index_ext"]:
                hints.append('👉 Point your index finger forward.')
            elif pointer_hand["middle_ext"] or pointer_hand["pinky_ext"] or pointer_hand["ring_ext"]:
                hints.append('👉 Curl your middle, ring, and pinky fingers into your palm.')

        # Check 3: Base hand posture
        target_pt = None

        if touch_target in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            target_pt = base_hand["tips"][touch_target]
            target_touch_coords = (float(target_pt[0]), float(target_pt[1]))
            # Base hand is open if it has >= 2 extended fingers, or target finger extended, or thumb spread
            base_open_ok = (base_hand["ext_count"] >= 2) or \
                           (touch_target == 'thumb' and base_hand["thumb_ext"]) or \
                           (np.linalg.norm(base_hand["tips"][touch_target] - base_hand["wrist"]) > base_hand["span"] * 0.60)

            checklist.append({
                'label': f'Base Hand Open ({touch_target.capitalize()} Target Ready)',
                'passed': bool(base_open_ok),
                'detail': f'Open base hand with {touch_target} visible.' if base_open_ok else f'Spread your fingers open with {touch_target} outstretched.'
            })
            score_components.append(25.0 if base_open_ok else 10.0)
            if not base_open_ok:
                hints.append(f'✋ Spread your {base_label_name} fingers open facing the camera.')
        elif touch_target == 'palm':
            target_pt = base_hand["pts"][9]  # Middle knuckle / palm center
            target_touch_coords = (float(target_pt[0]), float(target_pt[1]))
            checklist.append({
                'label': f'{base_label_name} Flat & Ready',
                'passed': True,
                'detail': 'Palm center ready as base.'
            })
            score_components.append(25.0)
        elif touch_target == 'loop':
            target_pt = (base_hand["tips"]["thumb"] + base_hand["tips"]["index"]) / 2.0
            target_touch_coords = (float(target_pt[0]), float(target_pt[1]))
            loop_ok = np.linalg.norm(base_hand["tips"]["thumb"] - base_hand["tips"]["index"]) < avg_span * 0.50
            checklist.append({
                'label': f'{base_label_name} Forming Circle Loop',
                'passed': bool(loop_ok),
                'detail': 'Thumb and index touching in circle.' if loop_ok else 'Touch thumb to index to make a loop.'
            })
            score_components.append(25.0 if loop_ok else 12.0)
            if not loop_ok:
                hints.append('👌 Touch thumb and index tips together to form a round circle.')
        elif touch_target == 'web':
            target_pt = (base_hand["pts"][2] + base_hand["pts"][5]) / 2.0  # Thumb-Index webbing
            target_touch_coords = (float(target_pt[0]), float(target_pt[1]))
            web_open = base_hand["thumb_ext"] and base_hand["index_ext"]
            checklist.append({
                'label': f'{base_label_name} Thumb-Index Webbing Open',
                'passed': bool(web_open),
                'detail': 'Thumb spread wide creating a V web.' if web_open else 'Spread thumb away from index finger.'
            })
            score_components.append(25.0 if web_open else 10.0)
            if not web_open:
                hints.append('👉 Spread your thumb wide to open the V webbing.')
        else:
            target_pt = base_hand["tips"]["thumb"]
            target_touch_coords = (float(target_pt[0]), float(target_pt[1]))
            score_components.append(25.0)

        # Check 4: Contact Distance to Target
        if target_pt is not None:
            # Measure distance to fingertip or distal joint/pad (e.g. ring finger tip 16 and DIP 15)
            cand_pts = [target_pt]
            if touch_target in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                dip_idx = {'thumb': 3, 'index': 7, 'middle': 11, 'ring': 15, 'pinky': 19}.get(touch_target)
                if dip_idx is not None:
                    cand_pts.append(base_hand["pts"][dip_idx])
            elif touch_target == 'loop':
                cand_pts.extend([base_hand["tips"]["thumb"], base_hand["tips"]["index"]])
            elif touch_target == 'index':
                cand_pts.append(base_hand["pts"][7])

            dist = min(np.linalg.norm(p_index_tip - pt) for pt in cand_pts) / avg_span
            touch_threshold = 0.58
            close_threshold = 0.98

            if dist < touch_threshold:
                checklist.append({
                    'label': f'Touching {guide.get("target_touch", "Target")}',
                    'passed': True,
                    'detail': f'Fingertips in contact! (Distance: {dist:.2f})'
                })
                score_components.append(25.0)
            elif dist < close_threshold:
                partial = max(5.0, 25.0 * (1.0 - (dist - touch_threshold) / (close_threshold - touch_threshold)))
                checklist.append({
                    'label': f'Touching {guide.get("target_touch", "Target")}',
                    'passed': False,
                    'detail': f'Almost there! Move pointer ~{int(dist * 8)} cm closer.'
                })
                score_components.append(partial)
                hints.append(f'📍 Move pointer finger closer to your {guide.get("target_touch", "base hand")}.')
            else:
                checklist.append({
                    'label': f'Touching {guide.get("target_touch", "Target")}',
                    'passed': False,
                    'detail': f'Hands too far apart (Distance: {dist:.2f}).'
                })
                score_components.append(0.0)
                hints.append(f'🎯 Touch your pointer finger to your {guide.get("target_touch", "base hand")}.')

    else:
        # SINGLE HAND EVALUATION (Numbers 0-5, Alright, Hello, etc.)
        hand = rh if rh is not None else lh
        hand_name = "Right Hand" if rh is not None else "Left Hand"
        ext_count = hand["ext_count"]
        exts = {
            'thumb': hand["thumb_ext"],
            'index': hand["index_ext"],
            'middle': hand["middle_ext"],
            'ring': hand["ring_ext"],
            'pinky': hand["pinky_ext"]
        }

        checklist.append({
            'label': f'{hand_name} Detected',
            'passed': True,
            'detail': f'{hand_name} actively tracked.'
        })
        score_components.append(30.0)

        if target_symbol == '0':
            fist_ok = (ext_count == 0)
            checklist.append({
                'label': 'All Fingers Curled (Fist)',
                'passed': bool(fist_ok),
                'detail': 'Clean closed fist.' if fist_ok else f'{ext_count} fingers still extended.'
            })
            score_components.append(70.0 if fist_ok else max(0.0, 70.0 - ext_count * 20.0))
            if not fist_ok:
                hints.append('✊ Curl all fingers and wrap thumb around into a tight fist.')

        elif target_symbol == '1':
            idx_ok = exts['index']
            others_curled = not (exts['middle'] or exts['ring'] or exts['pinky'] or exts['thumb'])
            checklist.append({'label': 'Index Finger Extended', 'passed': bool(idx_ok), 'detail': 'Index extended up.' if idx_ok else 'Extend index finger.'})
            checklist.append({'label': 'Other Fingers Curled', 'passed': bool(others_curled), 'detail': 'Clean single finger.' if others_curled else 'Fold other fingers into palm.'})
            score_components.append(35.0 if idx_ok else 0.0)
            score_components.append(35.0 if others_curled else 10.0)
            if not idx_ok:
                hints.append('☝️ Extend your index finger straight up.')
            if not others_curled:
                hints.append('👉 Curl your thumb, middle, ring, and pinky fingers into your palm.')

        elif target_symbol == '2':
            v_ok = exts['index'] and exts['middle']
            others_curled = not (exts['ring'] or exts['pinky'] or exts['thumb'])
            checklist.append({'label': 'Index & Middle Extended (V)', 'passed': bool(v_ok), 'detail': 'Both fingers up.' if v_ok else 'Extend both index and middle.'})
            checklist.append({'label': 'Ring, Pinky & Thumb Curled', 'passed': bool(others_curled), 'detail': 'Remaining fingers folded.' if others_curled else 'Fold ring & pinky into palm.'})
            score_components.append(35.0 if v_ok else 15.0)
            score_components.append(35.0 if others_curled else 10.0)
            if not v_ok:
                hints.append('✌️ Extend both your index and middle fingers.')
            if not others_curled:
                hints.append('👉 Fold your thumb over your ring and pinky fingers.')

        elif target_symbol == '5':
            all_open = (ext_count == 5)
            checklist.append({'label': 'All 5 Fingers Extended', 'passed': bool(all_open), 'detail': 'Full open hand.' if all_open else f'{ext_count}/5 fingers extended.'})
            score_components.append(70.0 * (ext_count / 5.0))
            if not all_open:
                hints.append('🖐️ Open and spread all 5 fingers wide facing the camera.')

        elif target_symbol == 'alright':
            thumb_up = exts['thumb']
            fingers_curled = not (exts['index'] or exts['middle'] or exts['ring'] or exts['pinky'])
            checklist.append({'label': 'Thumb Extended Upward', 'passed': bool(thumb_up), 'detail': 'Thumb clearly extended.' if thumb_up else 'Point thumb up.'})
            checklist.append({'label': 'Fingers Folded into Fist', 'passed': bool(fingers_curled), 'detail': 'Fingers curled into palm.' if fingers_curled else 'Curl remaining 4 fingers.'})
            score_components.append(35.0 if thumb_up else 0.0)
            score_components.append(35.0 if fingers_curled else 10.0)
            if not thumb_up:
                hints.append('👍 Point your thumb straight up in a thumbs-up sign.')
            if not fingers_curled:
                hints.append('👉 Curl your four fingers tightly into your palm.')

        elif target_symbol.lower() == 'hello':
            open_ok = (ext_count >= 4)
            at_temple = False
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                nose = pose_lms[0]
                # Hand is at or near temple/forehead level (above or level with nose/eyes)
                if hand["tips"]["index"][1] < nose.y + 0.06:
                    at_temple = True
            else:
                at_temple = True

            checklist.append({'label': 'Flat Open Hand (Salute B-Hand)', 'passed': bool(open_ok), 'detail': 'All fingers open flat.' if open_ok else 'Open all 5 fingers flat.'})
            checklist.append({'label': 'Raised to Temple / Forehead', 'passed': bool(at_temple), 'detail': 'Hand at temple/forehead level.' if at_temple else 'Raise hand higher to temple/forehead (at chin, it is "Thank You"!).'})
            score_components.append(35.0 if open_ok else 15.0)
            score_components.append(35.0 if at_temple else 5.0)
            if not open_ok:
                hints.append('🖐️ Open your hand flat with fingers together.')
            elif not at_temple:
                hints.append('📍 Raise your hand to your temple/forehead to sign "Hello" (at chin level, it becomes "Thank You"!).')

        elif target_symbol.lower() == 'thank you':
            open_ok = (ext_count >= 4)
            at_chin = False
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                nose = pose_lms[0]
                mouth_y = (pose_lms[9].y + pose_lms[10].y) / 2.0
                # Hand fingertips are near chin / lower lip level
                if hand["tips"]["index"][1] >= nose.y - 0.03 and hand["tips"]["index"][1] <= mouth_y + 0.16:
                    at_chin = True
            else:
                at_chin = True

            checklist.append({'label': 'Flat Hand with Fingers Together', 'passed': bool(open_ok), 'detail': 'Flat open hand.' if open_ok else 'Keep fingers flat together.'})
            checklist.append({'label': 'Fingertips at Chin / Lower Lip', 'passed': bool(at_chin), 'detail': 'Hand at chin level.' if at_chin else 'Touch fingertips to your chin/lower lip (at temple, it is "Hello"!).'})
            score_components.append(35.0 if open_ok else 15.0)
            score_components.append(35.0 if at_chin else 5.0)
            if not open_ok:
                hints.append('🖐️ Keep all fingers flat together.')
            elif not at_chin:
                hints.append('📍 Touch fingertips to your chin/lower lip, then move forward toward the viewer (at temple level, it becomes "Hello"!).')

        else:
            checklist.append({'label': 'Hand Active', 'passed': True, 'detail': 'Hand active in view.'})
            score_components.append(70.0)

    # Compute Total Accuracy Score
    total_acc = min(100.0, max(0.0, float(sum(score_components))))

    # Determine status & primary hint
    if total_acc >= 82.0:
        status = 'PERFECT'
        primary_hint = f'🌟 PERFECT "{target_symbol.upper()}"! Hold steady to master!'
    elif total_acc >= 55.0:
        status = 'GOOD'
        primary_hint = hints[0] if hints else '👍 Good form! Refine finger alignment.'
    else:
        status = 'NEEDS_WORK'
        primary_hint = hints[0] if hints else '💪 Follow the finger posture guide.'

    return {
        'accuracy': round(total_acc, 1),
        'status': status,
        'primary_hint': primary_hint,
        'checklist': checklist,
        'target_touch_coords': target_touch_coords,
        'pointer_coords': pointer_coords
    }


def draw_tutor_camera_overlay(image, eval_result, target_symbol):
    """
    Renders an interactive augmented coaching HUD on the OpenCV camera frame,
    including target touch crosshair, pointer guideline, accuracy meter, and guidance box.
    """
    h, w, _ = image.shape
    acc = eval_result['accuracy']
    status = eval_result['status']
    hint = eval_result['primary_hint']
    t_coords = eval_result.get('target_touch_coords')
    p_coords = eval_result.get('pointer_coords')

    # 1. Draw Target Touch Crosshairs and Pointer Connection
    if t_coords is not None:
        tx, ty = int(t_coords[0] * w), int(t_coords[1] * h)
        cv2.circle(image, (tx, ty), 16, (0, 215, 255), 2)
        cv2.circle(image, (tx, ty), 6, (0, 165, 255), -1)
        cv2.putText(image, "TARGET", (tx + 12, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

        if p_coords is not None:
            px, py = int(p_coords[0] * w), int(p_coords[1] * h)
            cv2.circle(image, (px, py), 8, (255, 105, 180), -1)
            cv2.line(image, (px, py), (tx, ty), (255, 200, 50), 1, cv2.LINE_AA)

    # 2. Top Header HUD Card
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.line(image, (0, 68), (w, 68), (70, 85, 105), 1)

    # Symbol Tag & Mode
    cv2.putText(image, f"PRACTICE: '{target_symbol.upper()}'", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Accuracy Score Badge
    badge_color = (0, 220, 90) if acc >= 80 else ((0, 180, 255) if acc >= 50 else (0, 60, 230))
    cv2.putText(image, f"ACCURACY: {int(acc)}%", (w - 180, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

    # Mini Progress Bar on Top
    bar_w = int((acc / 100.0) * (w - 30))
    cv2.rectangle(image, (15, 38), (w - 15, 48), (35, 45, 60), -1)
    cv2.rectangle(image, (15, 38), (15 + bar_w, 48), badge_color, -1)

    # Status subtext
    cv2.putText(image, f"Status: {status}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1)

    # 3. Bottom Coaching Advice Box
    bot_overlay = image.copy()
    cv2.rectangle(bot_overlay, (0, h - 50), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(bot_overlay, 0.82, image, 0.18, 0, image)
    cv2.line(image, (0, h - 50), (w, h - 50), (70, 85, 105), 1)

    # Coaching tip text
    clean_hint = hint.replace('👉', '>').replace('✋', '>').replace('☝️', '>').replace('✌️', '>').replace('✊', '>').replace('👍', '>').replace('🌟', '*').replace('⚠️', '!')
    cv2.putText(image, f"TUTOR: {clean_hint}", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return image
