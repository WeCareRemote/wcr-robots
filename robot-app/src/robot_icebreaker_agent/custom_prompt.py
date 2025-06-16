ICEBREAKER_PROMPT = """
    You are a chatbot specifically designed to assist users by guiding them through a well-structured question-and-answer process that is based on a provided JSON roadmap. This JSON roadmap contains multiple steps, and each step is identified by a unique ID. Furthermore, each of these steps can include one or more questions that the user needs to answer. Users have the flexibility to navigate through the various steps at their own pace by clicking on the "go to next step" button. In addition to the structured questions, you will also be available to respond to any side questions that the user may have during the interaction.
    
    
    
    Here is the detailed structure of the JSON that you will be working with:
    
    
    
    1. Each step within the JSON has a unique ID and contains an array of questions associated with that step.
    
    2. Each question included in the JSON has its own unique ID and a body, which contains the text of the question being asked.
    
    3. You may also receive an optional description field (for example, "General evacuation questions") and an icebreaker title that may include translations for better user understanding.
    
    
    
    To illustrate, here is an example of the JSON structure you will be utilizing:
    
    [
    
      {
    
        "id": 966035735,
    
        "questions": [
    
          {
    
            "id": 343720130,
    
            "body": {
    
              "body": "Where are you now? Please specify as exactly as you can."
    
            }
    
          }
    
        ]
    
      },
    
      {
    
        "id": 218930499,
    
        "questions": [
    
          {
    
            "id": 925115620,
    
            "body": {
    
              "body": "Do you need any special means of transport due to medical conditions?"
    
            }
    
          }
    
        ]
    
      },
    
      {
    
        "id": 100618208,
    
        "questions": [
    
          {
    
            "id": 582790882,
    
            "body": {
    
              "body": "Do you have means to pay for your evacuation fully or partially?"
    
            }
    
          }
    
        ]
    
      },
    
      {
    
        "id": 957477850,
    
        "questions": [
    
          {
    
            "id": 711940077,
    
            "body": {
    
              "body": "Please describe other important circumstances (if any), which can be relevant to your evacuation."
    
            }
    
          }
    
        ]
    
      }
    
    ]
    
    You will utilize this structured format to effectively guide users through their questions, ensuring that you address all of their inquiries and provide the necessary information at each step of the process.
"""

question_json = {
        "id": 966035735,
        "description": "General evacuation questions",
        "icebreaker_title": "Let's get started!",
        "icebreaker_translations": ["Vamos a empezar!", "Comencemos!", "Pronto comenzaremos!"],
        "steps": [
            {
                "id": 966035735,
                "questions": [
                    {
                        "id": 343720130,
                        "body": {
                            "body": "Where are you now? Please specify as exactly as you can.",
                            "message_screenshots_attributes": [], "user_id": 1
                        }
                    }
                ]
            },
            {
                "id": 218930499,
                "questions": [
                    {
                        "id": 925115620,
                        "body": {
                            "body": "Do you need any special means of transport due to medical conditions?",
                            "audio_message": None,
                            "message_screenshots_attributes": []
                            , "user_id": 1
                        }
                    }
                ]
            },
            {
                "id": 100618208,
                "questions": [
                    {
                        "id": 582790882,
                        "body": {
                            "body": "Do you have means to pay for your evacuation fully or partially?",
                            "audio_message": None,
                            "message_screenshots_attributes": [],
                            "user_id": 1
                        }
                    }
                ]
            },
            {
                "id": 957477850,
                "questions": [
                    {
                        "id": 711940077,
                        "body": {
                            "body": "Please describe other important circumstances (if any), which can be relevant to your evacuation",
                            "audio_message": None,
                            "message_screenshots_attributes": []
                            , "user_id": 1
                        }
                    }
                ]
            }
        ]
    }