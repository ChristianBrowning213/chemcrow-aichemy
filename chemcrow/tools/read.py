import os
import pandas as pd

from langchain.tools import BaseTool

class Read_CSV(BaseTool):
    name = "Read_CSV"
    description = "Input csv path, reads the csv file and gives the data inside csv file to the agent."

    def __init__(
        self,
    ):
        super().__init__()
        

    def _run(self, query: str):
        if os.path.exists(query):
            try:
                df = pd.read_csv(query)
                return df.head()
            except ValueError as e:
                return f"CSV parsing error: {e}"
            except Exception as e:
                return f"Error reading file: {e}"
        else:
            return "File not found or query is not a valid path"
            

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()