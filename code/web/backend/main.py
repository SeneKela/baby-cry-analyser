import sys
import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path to import from code/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference import predict_audio
from utils import interpret
from generate_report import create_report

app = FastAPI(title="CrySense AI API")

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "CrySense AI API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Development
        "https://senekela.github.io"  # Production GitHub Pages
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for uploads and reports
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

class AnalysisResponse(BaseModel):
    prediction: str
    confidence: float
    interpretation: str
    report_url: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    temp_filename = None
    converted_filename = None
    try:
        # Save uploaded file temporarily
        temp_filename = os.path.join(TEMP_DIR, f"temp_{file.filename}")
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Saved temp file: {temp_filename}")
        
        # Convert WebM to WAV if needed (torchaudio doesn't support WebM)
        if temp_filename.endswith('.webm'):
            try:
                from pydub import AudioSegment
                print("Converting WebM to WAV...")
                audio = AudioSegment.from_file(temp_filename, format="webm")
                converted_filename = temp_filename.replace('.webm', '_converted.wav')
                audio.export(converted_filename, format="wav")
                analysis_file = converted_filename
                print(f"Converted to: {converted_filename}")
            except ImportError:
                # If pydub not available, try direct conversion with subprocess
                import subprocess
                converted_filename = temp_filename.replace('.webm', '_converted.wav')
                try:
                    subprocess.run(['ffmpeg', '-i', temp_filename, '-ar', '16000', '-ac', '1', converted_filename], 
                                 check=True, capture_output=True)
                    analysis_file = converted_filename
                    print(f"Converted with ffmpeg: {converted_filename}")
                except:
                    raise Exception("WebM format not supported. Please install ffmpeg or pydub.")
        else:
            analysis_file = temp_filename
        
        # Run inference
        pred, conf, _ = predict_audio(analysis_file)
        print(f"Prediction: {pred}, Confidence: {conf}")
        
        # Get interpretation
        interpretation = interpret(pred, conf)
        
        # Generate report in temp directory
        pdf_filename = f"report_{file.filename.replace('.wav', '.txt').replace('.mp3', '.txt').replace('.webm', '.txt')}"
        pdf_path = os.path.join(TEMP_DIR, pdf_filename)
        
        # create_report returns the path, but we need to ensure it's in our temp dir
        create_report(pred, conf, file_in=file.filename)
        
        # Move the generated report to temp dir if it's not already there
        default_report_name = f"CrySense_Report_{file.filename.split('_')[-1] if '_' in file.filename else file.filename}".replace('.wav', '.txt').replace('.mp3', '.txt').replace('.webm', '.txt')
        # Find the most recent CrySense_Report file
        import glob
        reports = glob.glob("CrySense_Report_*.txt")
        if reports:
            # Get the most recent report
            latest_report = max(reports, key=os.path.getctime)
            shutil.move(latest_report, pdf_path)
        
        print(f"Report saved: {pdf_path}")
        
        # Clean up temp files
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        if converted_filename and os.path.exists(converted_filename):
            os.remove(converted_filename)
        
        return AnalysisResponse(
            prediction=pred,
            confidence=conf,
            interpretation=interpretation,
            report_url=f"/report/{os.path.basename(pdf_path)}"
        )
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        # Clean up temp files on error
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        if converted_filename and os.path.exists(converted_filename):
            os.remove(converted_filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{filename}")
async def get_report(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
