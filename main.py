import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File

from starlette.responses import RedirectResponse

from classifier import read_imagefile, predict

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/api/predict")
async def predict_images(file:UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction


# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000, reload=True)