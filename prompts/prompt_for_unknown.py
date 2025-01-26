import re, json, copy
import random
import math

# You are tasked with updating the conversations in landmark data to replace any Unknown landmark names with the provided ground truth (`landmark_name_gt`) (eg:a serene lake to U hájenky lake). Your goal is to adjust the conversation without changing the original content distribution. The modified conversations should reflect the specific landmark name while keeping the existing description and location knowledge intact.
# "conversations": "This image depicts a serene lake located in U_hájenky, surrounded by dense forests and a rural landscape. The calm water reflects the sky and surrounding trees, creating a peaceful and picturesque scene. Such lakes are often found in natural reserves or countryside areas, providing habitats for various wildlife and offering recreational opportunities for visitors. Lakes like this in U hájenky are commonly found in regions with abundant natural beauty, often serving as popular spots for fishing, boating, and hiking.\n\nThe lake in U_hájenky is valued for its tranquility and is often protected to preserve its natural state, making it an ideal location for nature enthusiasts and visitors.",
#   "landmark_name": "U hájenky lake"
# ### Instructions:
# 1. **Update Conversations and Landmark Name**: Modify the conversation to reflect the specific landmark name, update landmark_name with the landmark_name_gt 
# 2. **Keep conversation Intact**: Maintain the original conversation, including the description and location knowledge, but ensure it flows naturally with the updated landmark name.
# 3. **Example Output Structure**: The output should look like this:

# You are tasked with updating conversations in landmark data to replace any Unknown landmark names with the provided ground truth (landmark_name_gt) (e.g., "a serene lake" becomes "U hájenky lake"). Your goal is to adjust the conversation to incorporate the specific landmark name in the first sentence while modifying uncertain or vague expressions to reflect a more confident understanding, but without altering the original structure of the description. Ensure that the revised conversation retains the existing the existing description and location knowledge intact


PROMPT_UNKNOWN_LANDMARK="""
You are tasked with updating conversations in landmark data to replace any Unknown landmark names with the provided ground truth (landmark_name_gt) (e.g., "a serene lake" becomes "U hájenky lake"). Your goal is to adjust the conversation to incorporate the specific landmark name in the first sentence while modifying uncertain or vague expressions (e.g., "appears to be," "suggests it may have") to reflect a more confident and certain understanding of the landmark, based on known information. However, do not alter the overall structure of the original description. Ensure that the revised conversation retains the existing description and location knowledge intact, but with enhanced specificity and confidence where applicable.
### Input Data:
```json
{
  "conversations": {conversations},
  "landmark_name": {landmark_name},
  "landmark_name_gt": {landmark_name_gt}
}

### Example output:
```json
{
  "conversations": <updated_conversation>,
  "landmark_name": <landmark_name_gt>,
  "landmark_name_old": <landmark_name>,
  }
"""

def get_prompt_unknwon(item):
    prompt_text = PROMPT_UNKNOWN_LANDMARK.replace('{landmark_name_gt}',item[0]['landmark_name_gt']).replace('{landmark_name}', item[0]['landmark_name']).replace('{conversations}',item[0]['conversations'][-1]['value'])
    return prompt_text


def parse(item, response):
    if response is not None:
        output = json.loads(response['text'])
        response['text'] = output
        response['id'] = item['id']
        response['image'] = item['image']
        response['landmark_name'] = item['landmark_name']
    return response