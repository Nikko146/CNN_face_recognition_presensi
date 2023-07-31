import hashlib

from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
from functools import wraps
import mysql.connector
import re
import cv2
from PIL import Image
import numpy as np
import os
import time
import io
import xlwt
import base64

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# import tensorflow as tf
#
# # Load the trained model
# model = tf.keras.models.load_model('face_recognition_model.h5')
#
# # Get model summary
# model.summary()
#
# # Get model layer-wise details
# for layer in model.layers:
#     print(layer.name)
#     print(layer.get_config())
#     print(layer.get_weights())
#     print("------------")

app = Flask(__name__)

app.secret_key = 'poliban'
cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="ta_presensi"
)
mycursor = mydb.cursor()


# Definisi PILLOW, LBPH, numpy, cv2
# Cari yang include CNN

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Login / Logout >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged' in session:
            return f(*args, **kwargs)
        else:
            msg = "Diperlukan login terlebih dahulu !"
            return render_template('login.html', msg=msg)
    return wrap

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form :
        username = request.form['username']
        password = request.form['password']

        hash = password + app.secret_key
        hash = hashlib.md5(hash.encode())
        password = hash.hexdigest()

        mycursor.execute("SELECT * FROM users WHERE username = '{}' AND password = '{}'".format(username, password))
        account = mycursor.fetchone()

        if account :
            session['logged'] = True
            session['id'] = account[0]
            session['nama'] = account[3]

            return redirect(url_for('home'))
        else :
            msg = 'Username/Password Salah.'
    return render_template('login.html', msg=msg)

@app.route('/logout')
@login_required
def logout():
    msg = 'Logout Berhasil !'
    session.pop('logged', None)
    session.pop('id', None)
    session.pop('nama', None)

    return render_template('login.html', msg=msg)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(mahasiswa):
    face_classifier = cv2.CascadeClassifier(
        "C:/AI/Project/TA-Face-Recognition-Acara/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    mycursor.execute("SELECT IFNULL(MAX(id), 0) FROM img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    id = lastid
    max_imgid = id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + mahasiswa + "." + str(id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`id`, `id_mahasiswa`) VALUES
                                ('{}', '{}')""".format(id, mahasiswa))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/detail_kelas/train_classifier')
@login_required
def train_classifier():
    def load_dataset(dataset_dir):
        faces = []
        ids = []
        for face_file in os.listdir(dataset_dir):
            if face_file.endswith(".jpg"):
                face_path = os.path.join(dataset_dir, face_file)
                id = int(face_file.split(".")[0])
                face = Image.open(face_path).convert("L")
                face_array = np.array(face, dtype=np.uint8)
                faces.append(face_array)
                ids.append(id)
        return np.array(ids), np.array(faces)

    # Function to build the CNN model
    def build_cnn_model(input_shape, num_classes):
        model = Sequential()

        # cari penjelasan menggunakan nilai convolusi per layernya.
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    if __name__ == "__main__":
        dataset_dir = "C:/AI/Project/TA-Face-Recognition-Acara/dataset"
        input_shape = (200, 200, 1)  # Assuming the images are resized to (200, 200) grayscale images

        # Load the dataset
        ids, faces = load_dataset(dataset_dir)

        num_classes = len(ids) + 1  # Assuming you have 100 different student IDs (mahasiswa)

        # Normalize the images to values between 0 and 1
        faces = faces.astype('float32') / 255

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.2, random_state=42)

        # Reshape the data to fit the CNN model
        X_train = X_train.reshape(-1, 200, 200, 1)
        X_test = X_test.reshape(-1, 200, 200, 1)

        # Build the CNN model
        model = build_cnn_model(input_shape, num_classes)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Save the trained model
        model.save('face_recognition_model.h5')
        flash("Berhasil Training Data !")

    return redirect(url_for('kelas'))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Presensi Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition(mata_kuliah):
    # Function to preprocess the face images for the CNN model
    def preprocess_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))  # Assuming the CNN model expects input shape (200, 200, 1)
        img = img.reshape(1, 200, 200, 1)
        img = img.astype('float32') / 255.0
        return img

    # Function to load the CNN model for face recognition
    def load_cnn_model():
        return load_model("face_recognition_model.h5")

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf, occuring):
        # Perform face detection
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = faceCascade.detectMultiScale(gray_image, 1.1, 4)

        global justscanned
        global pause_cnt

        mycursor.execute("SELECT total_pertemuan, id_kelas FROM mata_kuliah WHERE id={}".format(mata_kuliah))
        row = mycursor.fetchone()

        pertemuan= row[0] + 1
        partisipasi_kelas = row[1]

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            face_roi = img[y:y + h, x:x + w]

            # Preprocess the face image for CNN model
            preprocessed_face = preprocess_image(face_roi)

            # Perform face recognition using the CNN model
            prediction = clf.predict(preprocessed_face)

            # Get id from prediction
            idPredict = np.argmax(prediction[0])

            # Confidence Prediction
            confPredict = prediction[0][idPredict]

            if confPredict > 0.7 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                occuring.append(idPredict)

                if int(cnt) == 30:
                    cnt = 0
                    idMostOccuring = max(set(occuring), key=occuring.count)

                    mycursor.execute("SELECT DISTINCT ds.id_mahasiswa, m.nama, m.id_kelas, k.nama_kelas"
                                     " FROM img_dataset ds "
                                     " LEFT JOIN mahasiswa m ON ds.id_mahasiswa = m.id "
                                     " LEFT JOIN kelas k ON m.id_kelas = k.id "
                                     " WHERE ds.id_mahasiswa = " + str(idMostOccuring))
                    row = mycursor.fetchone()

                    if row != None :
                        id_mata_kuliah = mata_kuliah
                        id_kelas = row[2]
                        nama_mahasiswa = row[1]
                        id_mahasiswa = row[0]
                        kelas_mahasiswa = row[3]

                        # Check sudah presensi per pertemuan
                        mycursor.execute("SELECT DISTINCT id_mahasiswa FROM presensi "
                                         "WHERE id_mahasiswa = {} AND id_mata_kuliah = {} AND pertemuan = {}"
                                         .format(id_mahasiswa, id_mata_kuliah, pertemuan))
                        sudah_presensi = mycursor.fetchall()

                        # Check partisipasi kelas
                        if id_kelas != partisipasi_kelas :
                            justscanned = "NotInKelas"
                            pause_cnt = 0
                            occuring.clear()
                        else :
                            if len(sudah_presensi) == 0:
                                mycursor.execute(
                                    "INSERT INTO presensi (id_mata_kuliah, id_mahasiswa, waktu_kehadiran, status, pertemuan) "
                                    "VALUES ('" + str(id_mata_kuliah) + "', '" + str(id_mahasiswa) + "', now(),'Hadir' ,'" + str(pertemuan) + "')")
                                mydb.commit()

                                cv2.putText(img, nama_mahasiswa + ' | ' + kelas_mahasiswa, (x - 10, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (153, 255, 255), 2, cv2.LINE_AA)
                                time.sleep(1)

                                justscanned = "Scanned"
                                pause_cnt = 0
                                occuring.clear()
                            elif len(sudah_presensi) >= 1:
                                justscanned = "Already"
                                pause_cnt = 0
                                occuring.clear()
                    else :
                        justscanned = "Unknown"
                        pause_cnt = 0
                        occuring.clear()
            else:
                if confPredict < 0.7 :
                    justscanned = "Unknown"

                if justscanned == "Already":
                    cv2.putText(img, 'Sudah Melakukan Presensi !', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                elif justscanned == "Scanned":
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                elif justscanned == "Unknown":
                    cv2.putText(img, 'Unknown', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                elif justscanned == "NotInKelas":
                    cv2.putText(img, 'Mahasiswa Bukan Dari Kelas Terkait', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade, occuring):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf, occuring)
        return img

    # Recognition Mulai Sini
    faceCascade = cv2.CascadeClassifier("C:/AI/Project/TA-Face-Recognition-Acara/resources/haarcascade_frontalface_default.xml")
    clf = load_cnn_model()

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

    occuring = []

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade, occuring)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Routing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Home Page >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/')
@login_required
def home():
    return render_template('index.html')

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Kelas >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/kelas')
@login_required
def kelas():
    mycursor.execute("SELECT * FROM kelas")
    data = mycursor.fetchall()

    return render_template('kelas/index.html', data=data)

@app.route('/tambah_kelas')
@login_required
def tambah_kelas():
    return render_template('kelas/tambah_kelas.html')

@app.route('/tambah_kelas_submit', methods=['POST'])
@login_required
def tambah_kelas_submit():
    nama_kelas = request.form.get('nama_kelas')
    mycursor.execute("""INSERT INTO `kelas` (`nama_kelas`) VALUES
                    ('{}')""".format(nama_kelas))
    mydb.commit()
    flash("Berhasil Menambahkan Data Kelas Baru !")

    return redirect(url_for('kelas'))

@app.route('/detail_kelas/<kelas>')
@login_required
def detail_kelas(kelas):
    # Get data kelas
    mycursor.execute("SELECT * FROM kelas WHERE id={}".format(kelas))
    kelas = mycursor.fetchone()

    # Get data mahasiswa in kelas
    mycursor.execute("SELECT * FROM mahasiswa WHERE id_kelas={} ORDER BY nim ASC".format(kelas[0]))
    mahasiswa = mycursor.fetchall()

    # Get count number of kelas participant
    mycursor.execute("SELECT COUNT(*) FROM mahasiswa WHERE id_kelas={}".format(kelas[0]))
    count = mycursor.fetchone()

    return render_template('kelas/detail_kelas.html', kelas=kelas, mahasiswa=mahasiswa, count=count)

@app.route('/edit_kelas/<kelas>')
@login_required
def edit_kelas(kelas):
    mycursor.execute("SELECT * FROM kelas WHERE id={}".format(kelas))
    data = mycursor.fetchone()
    return render_template('kelas/edit_kelas.html', data=data)

@app.route('/edit_kelas_submit', methods=['POST'])
@login_required
def edit_kelas_submit():
    id_kelas = request.form.get('id')
    nama_kelas = request.form.get('nama_kelas')
    mycursor.execute(""" UPDATE `kelas` SET `nama_kelas`='{}' WHERE id = {}""".format(nama_kelas, id_kelas))
    mydb.commit()

    # return redirect(url_for('home'))
    flash("Berhasil Mengubah Data Kelas !")
    return redirect(url_for('kelas'))

@app.route('/delete_kelas/<kelas>')
@login_required
def delete_kelas(kelas):
    mycursor.execute(""" DELETE FROM `kelas` WHERE id = {}""".format(kelas))
    mydb.commit()

    flash("Berhasil Menghapus Data Kelas !")

    # return redirect(url_for('home'))
    return redirect(url_for('kelas'))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Mahasiswa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/detail_kelas/<kelas>/tambah_mahasiswa')
@login_required
def tambah_mahasiswa(kelas):
    mycursor.execute("SELECT * FROM kelas WHERE id={}".format(kelas))
    data = mycursor.fetchone()

    return render_template('mahasiswa/tambah_mahasiswa.html', data=data)

@app.route('/detail_kelas/<kelas>/tambah_mahasiswa_submit', methods=['POST'])
@login_required
def tambah_mahasiswa_submit(kelas):
    id_kelas = request.form.get('id_kelas')
    nim = request.form.get('nim')
    nama = request.form.get('nama')
    jenis_kelamin = request.form.get('jenis_kelamin')
    alamat = request.form.get('alamat')
    mycursor.execute("""INSERT INTO `mahasiswa` (`id_kelas`, `nim`, `nama`, `jenis_kelamin`, `alamat`, `waktu_ditambahkan`) 
                            VALUES ('{}', '{}', '{}', '{}', '{}', now())""".format(id_kelas, nim, nama, jenis_kelamin, alamat))
    mydb.commit()
    flash("Berhasil Menambah Mahasiswa Baru !")

    return redirect(url_for('detail_kelas', kelas=id_kelas))

@app.route('/detail_kelas/<kelas>/detail_mahasiswa/<mahasiswa>')
@login_required
def detail_mahasiswa(kelas, mahasiswa):
    # Get data mahasiswa
    mycursor.execute("SELECT mahasiswa.*, kelas.* FROM mahasiswa LEFT JOIN kelas ON mahasiswa.id_kelas = kelas.id WHERE mahasiswa.id = {}".format(mahasiswa))
    data = mycursor.fetchone()

    # Check if dataset has been taken
    mycursor.execute("SELECT COUNT(*) FROM img_dataset WHERE id_mahasiswa='{}'".format(mahasiswa))
    check_dataset = mycursor.fetchone()

    if check_dataset[0] > 0:
        status = 'Sudah'
    else :
        status = 'Belum'

    return render_template('mahasiswa/detail_mahasiswa.html', mahasiswa=data, status=status)

@app.route('/detail_kelas/<kelas>/gen_dataset/<mahasiswa>')
@login_required
def gen_dataset(kelas, mahasiswa):
    return render_template('mahasiswa/generate_dataset_mahasiswa.html', kelas=kelas, mahasiswa=mahasiswa)

@app.route('/detail_kelas/<kelas>/video/<mahasiswa>')
@login_required
def video(kelas, mahasiswa):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(mahasiswa), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detail_kelas/<kelas>/edit_mahasiswa/<mahasiswa>')
@login_required
def edit_mahasiswa(kelas, mahasiswa):
    mycursor.execute("SELECT * FROM mahasiswa WHERE id={}".format(mahasiswa))
    data = mycursor.fetchone()

    # Get data for kelas dropdown
    mycursor.execute("SELECT * FROM kelas")
    dropdown_kelas = mycursor.fetchall()

    return render_template('mahasiswa/edit_mahasiswa.html', data=data, dropdown_kelas=dropdown_kelas)

@app.route('/detail_kelas/<kelas>/edit_mahasiswa_submit', methods=['POST'])
@login_required
def edit_mahasiswa_submit(kelas):
    id = request.form.get('id')
    nim = request.form.get('nim')
    nama = request.form.get('nama')
    jenis_kelamin = request.form.get('jenis_kelamin')
    alamat = request.form.get('alamat')
    id_kelas = request.form.get('id_kelas')
    mycursor.execute(""" UPDATE `mahasiswa` SET `nim`='{}', `nama`='{}', `jenis_kelamin`='{}', `alamat`='{}', `id_kelas`={} WHERE id = {}""".format(nim, nama, jenis_kelamin, alamat, id_kelas, id))
    mydb.commit()
    flash("Berhasil Mengubah Mahasiswa !")

    return redirect(url_for('detail_mahasiswa', kelas=kelas, mahasiswa=id))

@app.route('/detail_kelas/<kelas>/delete_mahasiswa/<mahasiswa>')
@login_required
def delete_mahasiswa(kelas, mahasiswa):
    mycursor.execute(""" DELETE FROM `mahasiswa` WHERE id = {}""".format(mahasiswa))
    mydb.commit()
    flash("Berhasil Menghapus Mahasiswa !")

    return redirect(url_for('detail_kelas', kelas=kelas))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Mahasiswa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/mata_kuliah')
@login_required
def mata_kuliah():
    mycursor.execute("SELECT mata_kuliah.*, kelas.nama_kelas FROM mata_kuliah "
                     "LEFT JOIN kelas ON mata_kuliah.id_kelas = kelas.id WHERE id_users = {}".format(session['id']))
    data = mycursor.fetchall()

    return render_template('mata_kuliah/index.html', data=data)

@app.route('/tambah_mata_kuliah')
@login_required
def tambah_mata_kuliah():
    mycursor.execute("SELECT * FROM kelas")
    dropdown_kelas = mycursor.fetchall()

    return render_template('mata_kuliah/tambah_mata_kuliah.html', dropdown_kelas=dropdown_kelas)

@app.route('/tambah_mata_kuliah_submit', methods=['POST'])
@login_required
def tambah_mata_kuliah_submit():
    nama_mata_kuliah = request.form.get('nama_mata_kuliah')
    id_users = session['id']
    id_kelas = request.form.get('id_kelas')
    jumlah_pertemuan = 16
    total_pertemuan = 0

    mycursor.execute("""INSERT INTO `mata_kuliah` (`nama_mata_kuliah`, `id_users`, `id_kelas`, `jumlah_pertemuan`, `total_pertemuan`) VALUES
                    ('{}', '{}', '{}', '{}', 'Belum Selesai')"""
                     .format(nama_mata_kuliah, id_users, id_kelas, jumlah_pertemuan, total_pertemuan))
    mydb.commit()
    flash("Berhasil Menambahkan Mata Kuliah Baru !")

    return redirect(url_for('mata_kuliah'))

@app.route('/detail_mata_kuliah/<mata_kuliah>')
@login_required
def detail_mata_kuliah(mata_kuliah):
    # Get data mata_kuliah
    mycursor.execute("SELECT mata_kuliah.*, kelas.nama_kelas FROM mata_kuliah "
                     "LEFT JOIN kelas ON mata_kuliah.id_kelas = kelas.id WHERE mata_kuliah.id={} ".format(mata_kuliah))
    mata_kuliah = mycursor.fetchone()

    return render_template('mata_kuliah/detail_mata_kuliah.html', mata_kuliah=mata_kuliah)

@app.route('/detail_mata_kuliah/<mata_kuliah>/detail_pertemuan/<pertemuan>')
@login_required
def detail_pertemuan_mata_kuliah(mata_kuliah, pertemuan):
    # Get data presensi
    mycursor.execute("SELECT presensi.* FROM presensi "
                     "LEFT JOIN mahasiswa ON mahasiswa.id = presensi.id_mahasiswa "
                     "WHERE presensi.pertemuan={} AND presensi.id_mata_kuliah = {} "
                     "ORDER BY presensi.id_mahasiswa ASC".format(pertemuan, mata_kuliah))
    data = mycursor.fetchall()

    # Get Information for mata_kuliah
    mycursor.execute("SELECT mata_kuliah.id, mata_kuliah.nama_mata_kuliah, kelas.nama_kelas, mata_kuliah.id_kelas FROM mata_kuliah "
                     "LEFT JOIN kelas ON kelas.id = mata_kuliah.id_kelas "
                     "WHERE mata_kuliah.id={}".format(mata_kuliah))
    info = mycursor.fetchone()

    # Get all kehadiran mahasiswa
    mycursor.execute("SELECT mahasiswa.id, mahasiswa.nama, mahasiswa.jenis_kelamin, presensi.status, presensi.waktu_kehadiran FROM presensi "
                     "LEFT JOIN mahasiswa ON mahasiswa.id = presensi.id_mahasiswa "
                     "WHERE presensi.pertemuan = {} AND presensi.id_mata_kuliah = {} "
                     "ORDER BY mahasiswa.id ASC".format(pertemuan, mata_kuliah))
    kehadiran = mycursor.fetchall()

    return render_template('mata_kuliah/pertemuan/detail_pertemuan_mata_kuliah.html', data=data, pertemuan=pertemuan, info=info, kehadiran=kehadiran)

@app.route('/update_status_kehadiran', methods=['POST'])
@login_required
def update_status_kehadiran():
    status = request.form.get('status')
    id_mahasiswa = request.form.get('id')
    id_mata_kuliah = request.form.get('id_mata_kuliah')
    pertemuan = request.form.get('pertemuan')

    mycursor.execute("UPDATE presensi SET status='{}', waktu_kehadiran=now() WHERE id_mahasiswa={} AND pertemuan={} AND id_mata_kuliah={}".format(status, id_mahasiswa, pertemuan, id_mata_kuliah))
    mydb.commit()

    return redirect(url_for('detail_pertemuan_mata_kuliah', mata_kuliah=id_mata_kuliah, pertemuan=pertemuan))


@app.route('/detail_mata_kuliah/<mata_kuliah>/detail_pertemuan/<pertemuan>/export')
def export_to_excel_per_pertemuan(mata_kuliah, pertemuan):
    # Get all kehadiran mahasiswa
    mycursor.execute(
        "SELECT mata_kuliah.nama_mata_kuliah, mahasiswa.id, mahasiswa.nama, mahasiswa.jenis_kelamin, presensi.status, presensi.waktu_kehadiran FROM presensi "
        "LEFT JOIN mahasiswa ON mahasiswa.id = presensi.id_mahasiswa "
        "LEFT JOIN mata_kuliah ON mata_kuliah.id = presensi.id_mata_kuliah "
        "WHERE presensi.pertemuan = {} AND presensi.id_mata_kuliah = {} "
        "ORDER BY mahasiswa.id ASC".format(pertemuan, mata_kuliah))
    row = mycursor.fetchall()

    # Get data sakit
    mycursor.execute(
        "SELECT COUNT(status) FROM presensi "
        "WHERE pertemuan = {} AND id_mata_kuliah = {} AND status='Sakit' ".format(pertemuan, mata_kuliah))
    jumlah_sakit = mycursor.fetchone()

    # Get data absen
    mycursor.execute(
        "SELECT COUNT(status) FROM presensi "
        "WHERE pertemuan = {} AND id_mata_kuliah = {} AND status='Absen' ".format(pertemuan, mata_kuliah))
    jumlah_absen = mycursor.fetchone()

    # Get data izin
    mycursor.execute(
        "SELECT COUNT(status) FROM presensi "
        "WHERE pertemuan = {} AND id_mata_kuliah = {} AND status='Izin' ".format(pertemuan, mata_kuliah))
    jumlah_izin = mycursor.fetchone()

    # Get data hadir
    mycursor.execute(
        "SELECT COUNT(status) FROM presensi "
        "WHERE pertemuan = {} AND id_mata_kuliah = {} AND status='Hadir' ".format(pertemuan, mata_kuliah))
    jumlah_hadir = mycursor.fetchone()

    # Get information about mata kuliah
    mycursor.execute(
        "SELECT mata_kuliah.nama_mata_kuliah, kelas.nama_kelas FROM mata_kuliah LEFT JOIN kelas ON kelas.id = mata_kuliah.id_kelas WHERE mata_kuliah.id={}".format(mata_kuliah))
    info = mycursor.fetchone()

    # output in bytes
    output = io.BytesIO()
    # create WorkBook object
    workbook = xlwt.Workbook()
    # add a sheet
    sh = workbook.add_sheet(info[0] +'_' + pertemuan)

    style_header = xlwt.Style.easyxf('font: bold on; align: horiz center')

    # add headers
    sh.write(0, 0, 'No', style=style_header)
    sh.write(0, 1, 'Nama Mahasiswa', style=style_header)
    sh.write(0, 2, 'Jenis Kelamin', style=style_header)
    sh.write(0, 3, 'Status Presensi', style=style_header)
    sh.write(0, 4, 'Waktu Kehadiran', style=style_header)
    sh.write(0, 7, 'Jumlah Hadir :', style=xlwt.Style.easyxf('font: bold on'))
    sh.write(1, 7, 'Jumlah Sakit :', style=xlwt.Style.easyxf('font: bold on'))
    sh.write(2, 7, 'Jumlah Izin :', style=xlwt.Style.easyxf('font: bold on'))
    sh.write(3, 7, 'Jumlah Absen :', style=xlwt.Style.easyxf('font: bold on'))


    idx = 0
    no = 1
    for data in row:
        sh.write(idx + 1, 0, no, style = xlwt.Style.easyxf('align: horiz center'))
        sh.write(idx + 1, 1, data[2], style = xlwt.Style.easyxf('align: horiz center'))
        sh.write(idx + 1, 2, data[3], style = xlwt.Style.easyxf('align: horiz center'))
        sh.write(idx + 1, 3, data[4], style = xlwt.Style.easyxf('align: horiz center'))
        sh.write(idx + 1, 4, str(data[5]), style = xlwt.Style.easyxf('align: horiz center'))

        idx += 1
        no += 1

    sh.write(0, 8, str(jumlah_hadir[0]), style=xlwt.Style.easyxf('align: horiz center'))
    sh.write(1, 8, str(jumlah_sakit[0]), style=xlwt.Style.easyxf('align: horiz center'))
    sh.write(2, 8, str(jumlah_izin[0]), style=xlwt.Style.easyxf('align: horiz center'))
    sh.write(3, 8, str(jumlah_absen[0]), style=xlwt.Style.easyxf('align: horiz center'))

    workbook.save(output)
    output.seek(0)

    return Response(output, mimetype="application/ms-excel",
                    headers={"Content-Disposition": "attachment;filename=rekap_"+ info[0] +"_"+ info[1] +"_pertemuan_"+ pertemuan +".xls"})

@app.route('/edit_mata_kuliah/<mata_kuliah>')
@login_required
def edit_mata_kuliah(mata_kuliah):
    # Get data for mata_kuliah Only
    mycursor.execute("SELECT * FROM mata_kuliah WHERE id={}".format(mata_kuliah))
    data = mycursor.fetchone()

    # Get data for kelas dropdown
    mycursor.execute("SELECT * FROM kelas")
    dropdown_kelas = mycursor.fetchall()

    return render_template('mata_kuliah/edit_mata_kuliah.html', data=data, dropdown_kelas = dropdown_kelas)

@app.route('/edit_mata_kuliah_submit', methods=['POST'])
@login_required
def edit_mata_kuliah_submit():
    id = request.form.get('id')
    nama_mata_kuliah = request.form.get('nama_mata_kuliah')
    id_kelas = request.form.get('id_kelas')

    mycursor.execute(""" UPDATE `mata_kuliah` SET `nama_mata_kuliah` ='{}', `id_kelas` ='{}' WHERE id = {}"""
                     .format(nama_mata_kuliah,id_kelas, id))
    mydb.commit()
    flash("Berhasil Mengubah Mata Kuliah !")

    return redirect(url_for('detail_mata_kuliah', mata_kuliah=id))

@app.route('/delete_mata_kuliah/<mata_kuliah>')
@login_required
def delete_mata_kuliah(mata_kuliah):
    mycursor.execute(""" DELETE FROM `mata_kuliah` WHERE id = {}""".format(mata_kuliah))
    mydb.commit()
    flash("Berhasil Menghapus Mata Kuliah !")

    return redirect(url_for('mata_kuliah', mata_kuliah=mata_kuliah))

@app.route('/detail_mata_kuliah/<mata_kuliah>/presensi')
@login_required
def presensi(mata_kuliah):
    mycursor.execute("SELECT p.id, p.id_mahasiswa, m.nama, m.nim, p.waktu_kehadiran "
                     " from presensi p "
                     " left join mahasiswa m on p.id_mahasiswa = m.id "
                     " where p.waktu_kehadiran = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('mata_kuliah/presensi/index.html', data=data, mata_kuliah=mata_kuliah)

@app.route('/detail_mata_kuliah/<mata_kuliah>/presensi/scan_wajah')
@login_required
def scan_wajah(mata_kuliah):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(mata_kuliah), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detail_mata_kuliah/<mata_kuliah>/presensi/load_data', methods=['GET', 'POST'])
@login_required
def loadData(mata_kuliah):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="ta_presensi"
    )
    mycursor = mydb.cursor()

    # Get data pertemuan
    mycursor.execute("SELECT total_pertemuan FROM mata_kuliah WHERE id={}".format(mata_kuliah))
    row = mycursor.fetchone()

    pertemuan_sekarang = row[0] + 1

    mycursor.execute("SELECT p.id, p.id_mahasiswa, m.nama, m.nim, k.nama_kelas, date_format(p.waktu_kehadiran, '%H:%i:%s') "
                     " FROM presensi p "
                     " LEFT JOIN mahasiswa m on p.id_mahasiswa = m.id "
                     " LEFT JOIN kelas k on m.id_kelas = k.id"
                     " WHERE p.id_mata_kuliah = {} AND p.pertemuan >= {}".format(mata_kuliah, pertemuan_sekarang))
    data = mycursor.fetchall()

    return jsonify(response=data)

@app.route('/detail_mata_kuliah/<mata_kuliah>/selesai')
@login_required
def selesai_presensi(mata_kuliah):
    # Get data total_pertemuan in mata_kuliah table
    mycursor.execute(""" SELECT total_pertemuan, id_kelas FROM mata_kuliah WHERE id={}""".format(mata_kuliah))
    row = mycursor.fetchone()

    total_pertemuan = row[0] + 1
    id_kelas = row[1]

    mycursor.execute(""" UPDATE `mata_kuliah` SET `total_pertemuan` ='{}' WHERE id = {}""".format(total_pertemuan, mata_kuliah))
    mydb.commit()

    # Get data mahasiswa tidak hadir
    mycursor.execute("SELECT * FROM mahasiswa "
                     "WHERE id "
                     "NOT IN (SELECT id_mahasiswa FROM presensi WHERE pertemuan={} AND id_mata_kuliah={}) "
                     "AND id_kelas = {}".format(total_pertemuan, mata_kuliah, id_kelas))
    row = mycursor.fetchall()

    for data in row :
        mycursor.execute("INSERT INTO presensi (id_mata_kuliah, id_mahasiswa, waktu_kehadiran, status, pertemuan) "
                         "VALUES ('" + str(mata_kuliah) + "', '" + str(data[0]) + "', NULL,'Absen' ,'" + str(total_pertemuan) + "')")
        mydb.commit()
    flash("Presensi Telah Selesai !")

    return redirect(url_for('detail_mata_kuliah', mata_kuliah=mata_kuliah))

@app.route('/pengguna')
@login_required
def pengguna():
    msg=''
    mycursor.execute("SELECT * FROM users")
    data = mycursor.fetchall()

    return render_template('users/index.html', data=data, msg=msg)

@app.route('/tambah_pengguna')
@login_required
def tambah_pengguna():
    msg=''
    return render_template('users/tambah_pengguna.html', msg=msg)

@app.route('/tambah_pengguna_submit', methods=['POST'])
@login_required
def tambah_pengguna_submit():
    status = False
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'nip' in request.form and 'nama' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        nama = request.form['nama']
        nip = request.form['nip']

        mycursor.execute('SELECT * FROM users WHERE username = "{}"'.format(username))
        account = mycursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash('User sudah ada pada sistem !')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username hanya dapat diisi karakter dan angka !')
        elif not username or not password or not nip or not nama:
            flash('Form harus diisi !')
        else:
            # Hash the password
            hash = password + app.secret_key
            hash = hashlib.md5(hash.encode())
            password = hash.hexdigest()
            # Account doesn't exist, and the form data is valid, so insert the new account into the accounts table
            mycursor.execute("INSERT INTO users VALUES (NULL, '{}', '{}', '{}', '{}')".format(username, password, nama, nip))
            mydb.commit()
            flash("Berhasil Menambahkan Pengguna Baru !")
            status = True
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Form Harus di Isi !')

    return render_template('users/tambah_pengguna.html', status=status)


@app.route('/delete_pengguna/<pengguna>')
@login_required
def delete_pengguna(pengguna):
    mycursor.execute(""" DELETE FROM `users` WHERE id = {}""".format(pengguna))
    mydb.commit()
    flash("Berhasil Menghapus Pengguna !")

    return redirect(url_for('pengguna'))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

