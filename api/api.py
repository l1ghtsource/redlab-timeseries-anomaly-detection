from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO

app = FastAPI()
