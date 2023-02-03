"""Utility functions related to the Leaderboard and associated REST API."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .task_configs import get_all_task_configs


class Session(requests.Session):
    """A requests session that automatically adds the API key to the headers."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    def get_task(self, task_name: str, raw: bool = False) -> dict:
        """Get the leaderboard for the task corresponding to task_name.

        Args:
            task_name (str): The name of the task.
            raw (bool): Whether to get the raw leaderboard or not.

        Returns:
            dict: A dictionary with the models for the task.

        Raises:
            ValueError: If the task is not found.
        """
        # Check if task is valid
        try:
            task_config = get_all_task_configs()[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found.")

        # Create endpoint, taking into account if we are getting raw data or not
        if raw:
            endpoint = f"{self.base_url}/{task_config.name}-raw"
        else:
            endpoint = f"{self.base_url}/{task_config.name}"

        # Get task from Leaderboard
        response = self.get(endpoint)

        # Check if we got a valid response and raise error if not
        if response.status_code == 204 or not response.headers[
            "content-type"
        ].strip().startswith("application/json"):
            raise ValueError(response.text)

        # Return the leaderboard
        task = response.json()

        # Check if we got a valid response
        if task == {"error": "Table not found"}:
            raise ValueError(f"Task {task_name} not found.")
        return task

    def get_model_for_task(
        self, task_name: str, model_id: str, raw: bool = False
    ) -> dict:
        """Get the entries on leaderboard for the model_id for the task corresponding to task_name.

        Args:
            task_name (str): The name of the task.
            model_id (str): The model id.
            raw (bool): Whether to get the raw leaderboard or not.

        Returns:
            dict: A dictionary with the model for the task.

        Raises:
            ValueError: If the task or model is not found.
        """
        # Check if task is valid
        try:
            task_config = get_all_task_configs()[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found.")

        # Create endpoint, taking into account if we are getting raw data or not
        if raw:
            endpoint = f"{self.base_url}/{task_config.name}-raw/{model_id}"
        else:
            endpoint = f"{self.base_url}/{task_config.name}/{model_id}"

        # Get the model from leaderboard
        response = self.get(endpoint)

        # Check if we got a valid response and raise error if not
        if response.status_code == 204 or not response.headers[
            "content-type"
        ].strip().startswith("application/json"):
            raise ValueError(response.text)

        task = response.json()

        # Check if we got a valid response
        if task == {"error": "Table not found"}:
            raise ValueError(f"Task {task_name} not found.")

        if task == {"error": "Model not found"}:
            raise ValueError(f"Model {model_id} not found.")
        return task

    def post_model_to_task(
        self, model_type: str, task_name: str, model_id: str, metrics: dict, test: bool
    ) -> dict:
        """Post a model to the leaderboard for the task corresponding to task_name.

        Args:
            model_type (str): The model type.
            task_name (str): The name of the task.
            model_id (str): The model id.
            metrics (dict): A dictionary with the metrics for the model.
            test (bool):
                Whether we are in test mode or not. If we are in test mode, we will not
                actually post the model to the leaderboard, but we will still return the response.

        Returns:
            dict: A dictionary with the possibly updated leaderboard.

        Raises:
            ValueError: If the task is not found, or if a non 200 response is returned from the API.
        """

        # Check if task is valid
        try:
            task_config = get_all_task_configs()[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found.")

        # Create endpoint
        endpoint = f"{self.base_url}/{task_config.name}"

        # Create payload
        payload = {
            "model_type": model_type,
            "task_name": task_name,
            "model_id": model_id,
            "metrics": metrics,
            "test": test,
        }

        # Post the model to leaderboard
        response = self.post(endpoint, json=payload)

        # Check if we got a valid response and raise error if not
        if response.status_code == 204 or not response.headers[
            "content-type"
        ].strip().startswith("application/json"):
            raise ValueError(response.text)

        # Return the leaderboard
        response_json = response.json()

        # Close the connection
        response.close()

        return response_json
