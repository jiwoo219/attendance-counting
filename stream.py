from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route("/")
def index() :
    return render_template('index.html')

def get_frame() :
    cap = cv2.VideoCapture(0)
    while True :
        _, frame = cap.read()
        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

    del(cap)

@app.route('/calc')
def calc() :
    return Response(get_frame(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__" :
    app.run(host="0.0.0.0", debug=True, threaded=True)