import re, json, copy
import random
import math
PROMPT_LANDMARK_V0 = '''{QUESTION} \n
[Reference Location]: {LOCATION} \n
[Note: The reference location provided may be incorrect. Please use your best judgment to identify the correct location and do not mention or rely on the reference location in your response.] \n
[Tasks: 1. **Entities**: List several entities that are strongly related to the landmark in the image. These entities should help distinguish the landmark and aid in its identification. \n
2. **Landmark**: Identify the exact location where the image was taken, specifying the name of the landmark. \n
3. **Answer**: Provide a detailed answer to the question. The answer should incorporate the identified entities and offer relevant information or knowledge about the landmark.] \n
[Output Format in JSON:{"entities": [],"landmark": "","answer": ""}]'''

PROMPT_LANDMARK_V1 = '''{QUESTION}
[Reference Location]: {LOCATION}
[Your response must adhere to the following requirements:
1. In addition to identifying the location where the photo was taken, you must provide a detailed description of the photo. This should include specific features visible in the image, such as architectural elements, landscape features, or any notable details. Moreover, please provide some relevant information or knowledge about the location, including its historical significance, cultural context, and any unique aspects that make the landmark important. Do not mention or refer to the reference location in your response. 
2. List several entities that are strongly related to the landmark in the image. These entities should be integral to the landmark's identification, such as distinctive architectural features, materials, designers, historical events, or cultural elements associated with the location.]
[Output Format in JSON:{"entities": [],"landmark": "","answer": ""}]'''


PROMPT_LANDMARK_V2 ='''
You are tasked with labeling a batch of images depicting landmarks from around the world. You will be provided with a reference name of the landmark.
[Reference Location]: {LOCATION}
[Note: The reference location provided may be incorrect. Please use your best judgment to identify the correct location and do not mention or rely on the reference location in your response.] \n
Your task is to generate rich content that includes:
1. **Entities Identification**:
   - Identify and list specific entities that can help recognize the landmark in the image but don't directly mention the landmark name. These entities should include both visual features (e.g., structural elements, colors, shapes) and conceptual features understood by a language model (e.g., architectural style, historical significance, cultural context, environment).
2. **Landmark Description**:
   - Provide a concise description of the landmark. This should include key information about its appearance, purpose, and historical or cultural significance. The description should be informative enough to offer context to someone unfamiliar with the landmark.
3. **Location Knowledge**:
   - Share additional knowledge related to the landmark's location. This could include details about the environment surrounding the landmark, notable facts about the area, or any relevant historical or cultural anecdotes.

### Example Output Structure:
```json
{
  "landmark_name": "Eiffel Tower",
  "entities": [
    "wrought iron lattice structure",
    "Parisian skyline",
    "Gustave Eiffel",
    "19th-century engineering",
    "tourist attraction",
    "World's Fair 1889"
  ],
  "description": "The Eiffel Tower is a wrought iron lattice tower in Paris, France, named after the engineer Gustave Eiffel. It was constructed for the 1889 World's Fair and has since become a global icon of France, known for its unique structure and panoramic views of Paris.",
  "location_knowledge": "The Eiffel Tower is located on the Champ de Mars near the Seine River in Paris. The surrounding area is a popular destination for tourists and locals alike, offering scenic views, boat tours along the Seine, and proximity to other Parisian landmarks such as the Louvre and Notre-Dame Cathedral."
}
'''

PROMPT_LANDMARK_PART ='''
You are tasked with labeling a batch of images depicting landmarks from around the world. You will be provided with a reference name of the landmark.
[Reference Location]: {LOCATION}
[Note: The reference location provided may be incorrect. Please use your best judgment to identify the correct location. If you are unable to confidently identify the correct location, you may rely on the provided reference location.] \n
Your task is to generate rich content that includes:
1. **Landmark Description**:
   - Provide a concise description of the landmark. This should include key information about its appearance, purpose, and historical or cultural significance. The description should be informative enough to offer context to someone unfamiliar with the landmark.
2. **Location Knowledge**:
   - Share additional knowledge related to the landmark's location. This could include details about the environment surrounding the landmark, notable facts about the area, or any relevant historical or cultural anecdotes.

### Example Output Structure:
```json
{
  "landmark_name": <landmark_name>,
  "description": <description> ,
  "location_knowledge": <location_knowledge>,
}
'''

# ```json
# {
#   "landmark_name": "Eiffel Tower",
#   "description": "The Eiffel Tower is a wrought iron lattice tower in Paris, France, named after the engineer Gustave Eiffel. It was constructed for the 1889 World's Fair and has since become a global icon of France, known for its unique structure and panoramic views of Paris.",
#   "location_knowledge": "The Eiffel Tower is located on the Champ de Mars near the Seine River in Paris. The surrounding area is a popular destination for tourists and locals alike, offering scenic views, boat tours along the Seine, and proximity to other Parisian landmarks such as the Louvre and Notre-Dame Cathedral."
# }
# '''

PROMPT_QA = '''{QUESTION} 
[Reference Location]: {LOCATION} 
[Note: The reference location may be incorrect, so you need to rely on your own ability to answer.]
[Your response must adhere to the following requirements]: 
1. The response must be in English; 
2. Besides identifying the location where the photo was taken, you should also describe the photo and share some knowledge related to the location; 
3. Do not mention the reference location or this note in your response.'''



PROMPT_LANDMARK_SUPERVISE ='''
You are tasked with labeling a batch of images depicting landmarks from around the world. You will be provided with the name and description of the landmark in the image.
[Location]: {LOCATION}
[DESCRIPTION]: {description}
Using both the image and the description provided, your task is to generate rich content that includes:
1. **Entities Identification**:
   - Identify and list specific entities that can help recognize the landmark. These entities should include both visual features (e.g., structural elements, colors, shapes) and conceptual features understood by a language model (e.g., architectural style, historical significance, cultural context, environment) but avoid explicitly using the landmark name.
2. Given the original hierarchical label: '{hierarchical_label}'(may be unknown and incorrect), return the most (only return one) related new hierarchical label from the following list:[
    "mountain", "volcano", "lake", "waterfall", "river", "wetland", "ocean area", 
    "beach", "cliff", "cave", "island", "tree", "botanical garden", "parks", "trail", 
    "agricultural land", "stone", "canyon", "desert", "fjord", "glacier", "peninsula", 
    "biosphere reserve", "well", "salt flat", "church", "temple", "castle / fort", 
    "palace", "monastery", "tower", "memorial", "cemetery", "ruins", "pyramid", 
    "winery", "sports venue", "road", "museum", "theatre", "library", "zoo", "shopping", 
    "house", "square", "hotel", "restaurant", "school", "hospital", "prison", "embassy", 
    "concert hall", "town hall", "cinema", "fountain", "post office", "bank", "gate", 
    "bridge", "lighthouse", "harbor", "canal", "mine", "dam", "factory", "power plant", 
    "air transportation", "rail transportation", "cable transportation", "aqueduct", 
    "tunnel", "ship", "market", "sculpture", "artwork", "bath", "swimming pool", 
    "amusement park", "windmill", "stairs", "observatory", "skyscraper", "conference center",
    "casino","festival","aquarium"]

### Example Output Structure for Eiffel Tower:
```json
{
   "entities": [
    "wrought iron lattice structure",
    "Parisian skyline",
    "Gustave Eiffel",
    "19th-century engineering",
    "tourist attraction",
    "World's Fair 1889"
  ],
   "new_hierarchical_label": "tower",
   "if_new_hierarchical_label_different": "false"
  }
'''


PROMPT_LANDMARK_ALL ='''
You are tasked with labeling a batch of images depicting landmarks from around the world. You will be provided with a reference name and the original hierarchical label of the landmark.
[Reference Location]: {LOCATION}
[Hierarchical Label]: {hierarchical_labels}
[Note: The reference location provided may be incorrect. Please use your best judgment to identify the correct location. If you are unable to confidently identify the correct location, you may rely on the provided reference location.] \n
Your task is to generate rich content that includes:

1. **Entities Identification**:
   - Identify and list specific entities that can help recognize the landmark in the image but don't directly mention the landmark name. These entities should include both visual features (e.g., structural elements, colors, shapes) and conceptual features understood by a language model (e.g., architectural style, historical significance, cultural context, environment).
2. **Landmark Description**:
   - Provide a concise description of the landmark. This should include key information about its appearance, purpose, and historical or cultural significance. The description should be informative enough to offer context to someone unfamiliar with the landmark.
3. **Location Knowledge**:
   - Share additional knowledge related to the landmark's location. This could include details about the environment surrounding the landmark, notable facts about the area, or any relevant historical or cultural anecdotes.
4. **Hierarchical Label**:
   - Given the refernce hierarchical label(may be unknown), return the most (only return one) related new hierarchical label from the following list:[
    "mountain", "volcano", "lake", "waterfall", "river", "wetland", "ocean area", 
    "beach", "cliff", "cave", "island", "tree", "botanical garden", "parks", "trail", 
    "agricultural land", "stone", "canyon", "desert", "fjord", "glacier", "peninsula", 
    "biosphere reserve", "well", "salt flat", "church", "temple", "castle / fort", 
    "palace", "monastery", "tower", "memorial", "cemetery", "ruins", "pyramid", 
    "winery", "sports venue", "road", "museum", "theatre", "library", "zoo", "shopping", 
    "house", "square", "hotel", "restaurant", "school", "hospital", "prison", "embassy", 
    "concert hall", "town hall", "cinema", "fountain", "post office", "bank", "gate", 
    "bridge", "lighthouse", "harbor", "canal", "mine", "dam", "factory", "power plant", 
    "air transportation", "rail transportation", "cable transportation", "aqueduct", 
    "tunnel", "ship", "market", "sculpture", "artwork", "bath", "swimming pool", 
    "amusement park", "windmill", "stairs", "observatory", "skyscraper", "conference center",
    "casino","festival","aquarium"].

### Example Output Structure:
```json
{
  "landmark_name": "Eiffel Tower",
  "entities": [
    "wrought iron lattice structure",
    "Parisian skyline",
    "Gustave Eiffel",
    "19th-century engineering",
    "tourist attraction",
    "World's Fair 1889"
  ],
  "description": "The Eiffel Tower is a wrought iron lattice tower in Paris, France, named after the engineer Gustave Eiffel. It was constructed for the 1889 World's Fair and has since become a global icon of France, known for its unique structure and panoramic views of Paris.",
  "location_knowledge": "The Eiffel Tower is located on the Champ de Mars near the Seine River in Paris. The surrounding area is a popular destination for tourists and locals alike, offering scenic views, boat tours along the Seine, and proximity to other Parisian landmarks such as the Louvre and Notre-Dame Cathedral.",
  "new_hierarchical_label": "tower"
}
'''


QUESTIONS = [
    'Where was this photo taken?',
    'Identify the location where this photo was taken.',
    'What is the location shown in the image?',
    'Tell me where this photo was taken.',
    'Where might this photo have been taken?',
    'What place is depicted in this image?'
]


def get_prompt_supervise(item):
   hierarchical_label = item['hierarchical_label']
   if hierarchical_label is None or (isinstance(hierarchical_label, float) and math.isnan(hierarchical_label)):
      hierarchical_label = 'Unknown'
   desc = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt')
   prompt_text = PROMPT_LANDMARK_SUPERVISE.replace('{QUESTION}', random.choice(QUESTIONS)).replace('{LOCATION}', item['landmark_name']).replace('{hierarchical_label}', hierarchical_label).replace('{description}',desc)
   return prompt_text

def get_prompt_all(item):

   hierarchical_label = item['hierarchical_label']
   if hierarchical_label is None or (isinstance(hierarchical_label, float) and math.isnan(hierarchical_label)):
      hierarchical_label = 'Unknown'
   prompt_text = PROMPT_LANDMARK_ALL.replace('{QUESTION}', random.choice(QUESTIONS)).replace('{LOCATION}', item['label']).replace('{hierarchical_labels}',hierarchical_label)
   return prompt_text

def get_prompt_part(item):
   import pdb;pdb.set_trace()
   prompt_text = PROMPT_LANDMARK_PART.replace('{QUESTION}', random.choice(QUESTIONS)).replace('{LOCATION}', item['label'])
   return prompt_text

def get_prompt(label):
    prompt_text = PROMPT_LANDMARK.replace('{QUESTION}', random.choice(QUESTIONS)).replace('{LOCATION}', label)
    return prompt_text

def parse(response):
    # Remove the markdown json code block markers ```json ... ```
    response_all = response
    parsed_data = response_all 
    response_cleaned = re.sub(r'^```json|```$', '', response_all['text'].strip(), flags=re.MULTILINE)
    try:
        response_json = json.loads(response_cleaned)
        parsed_data['text'] = response_json
    except json.JSONDecodeError:
        parsed_data = {}
        parsed_data['text'] = "Failed to decode JSON response"
    return parsed_data
