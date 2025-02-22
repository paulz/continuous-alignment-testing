import json
import os
from tests.settings import ROOT_DIR
from openai import OpenAI


def test_allocations():
    client = OpenAI()
    assert client is not None
    json_path = os.path.join(ROOT_DIR, "fixtures", "skills.json")
    with open(json_path, "r") as file:
        skills_data = json.load(file)

    json_path = os.path.join(ROOT_DIR, "fixtures", "example_output.json")
    with open(json_path, "r") as file:
        example_output = json.load(file)

    acceptable_people = ["Sam Thomas", "Drew Anderson", "Alex Wilson", "Alex Johnson"]

    system_prompt = f"""
        You will get a description of a project, and your task is to tell me the best developers from the given list for the project
         based on their skills.
        Today's date is April 15th, 2025.
        Pick only developers who are available after the project start date. Pick people with higher skill levels first.
        respond in json with this structure:
            {example_output}

        Here is the skills data:
        """
    system_prompt = system_prompt + str(skills_data)

    project_description = """
        This is a mobile project for telecommunication company. The project starts June 3rd.
        It will find exciting moments from sports highlights videos.
        """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": project_description},
        ],
        response_format={"type": "json_object"},
    )
    response = completion.choices[0].message.content
    person_with_relevant_skill_was_selected = any(
        name in response for name in acceptable_people
    )
    assert person_with_relevant_skill_was_selected

