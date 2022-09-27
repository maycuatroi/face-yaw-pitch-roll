from insightface.app import FaceAnalysis

app = FaceAnalysis(
    allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider"]
)
reg_model = app.models["recognition"]
app.prepare(ctx_id=0, det_size=(640, 640))
