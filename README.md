# IA
IA profesor Alcaraz

# Pimer Commit 2/14/2024 
## Secuencia de percepción y Medida de rendimiento.
### Flavio joséfo
```
Secuencia de percepción:
-Comprender el problema y su lógica para poder desarrollar un algoritmo eficiente y óptimizado.
-Establecer el número definido de personas que conformarán el círculo.
-Saber el patrón en el cuál cafa persona matará a la otra.
-Desarrollar el algoritmo que permita, con base a lo anterior, saber qué persona quedará al último.

Medida de rendimiento:
-La posición que cada persona debe tomar en el círuclo.
-Saber qué persona quedará viva al último.
```
### Problema estrella: 
```
Secuencia de percepción:
-Comprender la matriz que se utiliza a la hora de comprender el problema.
-Estudiar el problema para desarrollar un algortimo que nos permita encontrar la mejor solución posible.
-Dividir la matriz en cruz para utilizar los 4 cuadrantes.
-Llenar cada cuadrante a la vez.
-El primer reo deberá ir al cuadrante inferior derecho en la ezquina superior izquierda y de ahí empezar a llenar fila por fila de ese cuadrante.
-Una vez terminado el cuadrante el siguiente reo (#26) deberá empezar de igual forma pero a la inversa el cuadrante inferior izquierdo.

Medida de rendimiento:
-Optimizar las oportunidades para encontrar su número de celda.
-Desarrollar una estrategia que les permita a los reos llegar a su celda.
-La comunicación antes de entrar entre los reos.
```
# Reconocimiento facial
## Captura de rostro
#### En este script lo que realizamos es la captira de los rostros con ayuda de la librería CV2 de python, la cuál nos permite hacer conexión con la cámara de la computadora
#### Hice uso de la técnica de Haarcasade para poder identificar qué era un rostro y, con ayuda de la librería, poder resaltarlo en en cuadro vede y tomar un recorte de ese frame
#### para posteriormente guardarlo y almacenarlo en la computadora para poder generar el archivo XML con el cual trabajaría más adelante.
```
import numpy as np 
import cv2 as cv

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
def escala(imx, escala):
    width = int(imx.shape[1] * escala / 100)
    height = int(imx.shape[0] * escala / 100)
    size = (width, height)
    im = cv.resize(imx, size, interpolation = cv.INTER_AREA)
    return im

cap = cv.VideoCapture(0)
i = 0 
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
        #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        frame2 = frame[y:y+h, x:x+w]
        cv.imshow('rostros2', frame2)
        frame2 = cv.resize(frame2, (100,100), interpolation = cv.INTER_AREA)
        cv.imwrite('C:/Users/omar_/OneDrive/Documentos/IA/caras/Omar/img'+str(i)+'.png', frame2)
        
    cv.imshow('rostros', frame)
    i=i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```
## Creación de XML 
```
import cv2 as cv 
import numpy as np 
import os

dataSet = "C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\caras"
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet+'\\'+face
    print(f"face: {face}")
    print(f"PATH: {facePath}")
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'\\'+faceName,0))
    label = label + 1
print(np.count_nonzero(np.array(labels)==0)) 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('Eigenface.xml')
```
## Reconocimiento final 
```
import cv2 as cv
import os 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.read('Eigenface.xml')
dataSet = "C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\caras"
faces  = os.listdir(dataSet)

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        #cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (0,0,0), 1, cv.LINE_AA)
        if result[1] > 2800:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```

# Reconocimiento Waldo
## RecorteImagenWaldo
```
from PIL import Image
import os

def recortar_imagen(imagen_path, carpeta_destino, tamaño_recorte):
    imagen = Image.open(imagen_path)
    #Obtienemos las dimensiones de la imagen
    ancho, alto = imagen.size
    i = 134

    #Itera a través de la imagen y recorta en secciones de tamaño_recorte x tamaño_recorte
    for y in range(0, alto, tamaño_recorte):
        for x in range(0, ancho, tamaño_recorte):
            #Defino las coordenadas de la región para realizar recorte
            x_final = min(x + tamaño_recorte, ancho)
            y_final = min(y + tamaño_recorte, alto)
            box = (x, y, x_final, y_final)

            recorte = imagen.crop(box)
            #Guarda losr recortes en la carpeta destino
            i+=1
            nombre_archivo = f"recorte_{i}.png"
            recorte.save(os.path.join(carpeta_destino, nombre_archivo))

# Ruta de la imagen
imagen_path = "C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\Fondos\\Fondo2.jpg"  # Cambia "imagen.jpg" por la ruta de tu imagen
# Carpeta destino para guardar los recorte
carpeta_destino = "C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\FondosRecortados"  # Nombre de la carpeta donde guardar los recortes
# Tamaño del recorte
tamaño_recorte = 50
# Crea la carpeta si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)
    # Llama a la función para recortar la imagen
recortar_imagen(imagen_path, carpeta_destino, tamaño_recorte)
print("Recortes guardados en la carpeta:", carpeta_destino)
```
## Reescalado de imagenes
```
import cv2 as cv
import os
import numpy as np

def rota(img, i):
    rotacion = 0
    while rotacion < 360:
        h,w =  img.shape[:2]
        mw = cv.getRotationMatrix2D((h//2, w//2),rotacion,0.5)
        img2 = cv.warpAffine(img,mw,(h,w))
        cv.imwrite('C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\rotadas\\waldoR'+str(rotacion)+'-'+str(i)+'.jpg',img2)
        rotacion += 10

def escalaGris(img, i):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imwrite('C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\Gris'+'-'+str(i)+'.jpg',gray)

def escalaImagen(img,i):
    img2 = ''

i = 0
imgPaths = "C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\waldo"
nomFiles = os.listdir(imgPaths)
for nomFile in nomFiles:
    i = i + 1
    #print(f"Path: {imgPaths}")
    #print(f"nomFile: {nomFiles}")
    imgPath = imgPaths+"\\"+nomFile
    #print(f"imPath:  {imgPath}")
    img = cv.imread(imgPath)
    #print(f"IMG: {img}")
    rota(img, i)
```
## Reconocimiento final
```
import cv2 as cv 

rostro = cv.CascadeClassifier('C:\\Users\\omar_\\OneDrive\\Documentos\\IA\\dataset\\classifier\\stage0.xml')
cap = cv.VideoCapture(0)

i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
       frame2 = frame[ y:y+h, x:x+w]
        #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
       #cv.imwrite('/home/likcos/pruebacaras/juan/juan'+str(i)+'.jpg', frame2)
       cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```

# Juego Phaser V1
```
var w=800;
var h=400;
var jugador;
var fondo;

var bala, balaD=false, nave;

var salto;
var menu;

var velocidadBala;
var despBala;
var estatusAire;
var estatuSuelo;

var nnNetwork , nnEntrenamiento, nnSalida, datosEntrenamiento=[];
var modoAuto = false, eCompleto=false;



var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render:render});

function preload() {
    juego.load.image('fondo', 'assets/game/fondo.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png',32 ,48);
    juego.load.image('nave', 'assets/game/ufo.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');

}



function create() {

    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 800;
    juego.time.desiredFps = 30;

    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
    nave = juego.add.sprite(w-100, h-70, 'nave');
    bala = juego.add.sprite(w-100, h, 'bala');
    jugador = juego.add.sprite(50, h, 'mono');


    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    var corre = jugador.animations.add('corre',[8,9,10,11]);
    jugador.animations.play('corre', 10, true);

    juego.physics.enable(bala);
    bala.body.collideWorldBounds = true;

    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, self);
    juego.input.onDown.add(mPausa, self);

    salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);

    
    nnNetwork =  new synaptic.Architect.Perceptron(2, 6, 6, 2);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);

}

// Función para exportar los datos a un archivo CSV
function exportarCSV() {
    var variables = [despBala, velocidadBala, estatusAire, estatuSuelo];
    var csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Desplazamiento Bala,Velocidad Bala,Estatus Aire,Estatus Suelo\r\n";
    csvContent += variables.join(",") + "\r\n";

    var encodedUri = encodeURI(csvContent);
    var link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "datos_juego.csv");
    document.body.appendChild(link);
    link.click();
}

// Asigna la función de exportación al evento clic del botón
document.getElementById("exportButton").addEventListener("click", function() {
    exportarCSV();
});

function enRedNeural(){
    nnEntrenamiento.train(datosEntrenamiento, {rate: 0.0003, iterations: 10000, shuffle: true});
}


function datosDeEntrenamiento(param_entrada){

    console.log("Entrada",param_entrada[0]+" "+param_entrada[1]);
    nnSalida = nnNetwork.activate(param_entrada);
    var aire=Math.round( nnSalida[0]*100 );
    var piso=Math.round( nnSalida[1]*100 );
    console.log("Valor ","En el Aire %: "+ aire + " En el suelo %: " + piso );
    return nnSalida[0]>=nnSalida[1];
}



function pausa(){
    juego.paused = true;
    menu = juego.add.sprite(w/2,h/2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event){
    if(juego.paused){
        var menu_x1 = w/2 - 270/2, menu_x2 = w/2 + 270/2,
            menu_y1 = h/2 - 180/2, menu_y2 = h/2 + 180/2;

        var mouse_x = event.x  ,
            mouse_y = event.y  ;

        if(mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2 ){
            if(mouse_x >=menu_x1 && mouse_x <=menu_x2 && mouse_y >=menu_y1 && mouse_y <=menu_y1+90){
                eCompleto=false;
                datosEntrenamiento = [];
                modoAuto = false;
            }else if (mouse_x >=menu_x1 && mouse_x <=menu_x2 && mouse_y >=menu_y1+90 && mouse_y <=menu_y2) {
                if(!eCompleto) {
                    console.log("","Entrenamiento "+ datosEntrenamiento.length +" valores" );
                    enRedNeural();
                    eCompleto=true;
                }
                modoAuto = true;
            }

            menu.destroy();
            resetVariables();
            juego.paused = false;

        }
    }
}


function resetVariables(){
    jugador.body.velocity.x=0;
    jugador.body.velocity.y=0;
    bala.body.velocity.x = 0;
    bala.position.x = w-100;
    jugador.position.x=50;
    balaD=false;
}


function saltar(){
    jugador.body.velocity.y = -270;
}


function update() {

    fondo.tilePosition.x -= 1; 

    juego.physics.arcade.collide(bala, jugador, colisionH, null, this);

    estatuSuelo = 1;
    estatusAire = 0;

    if(!jugador.body.onFloor()) {
        estatuSuelo = 0;
        estatusAire = 1;
    }
	
    despBala = Math.floor( jugador.position.x - bala.position.x );

    if( modoAuto==false && salto.isDown &&  jugador.body.onFloor() ){
        saltar();
    }
    
    if( modoAuto == true  && bala.position.x>0 && jugador.body.onFloor()) {

        if( datosDeEntrenamiento( [despBala , velocidadBala] )  ){
            saltar();
        }
    }

    if( balaD==false ){
        disparo();
    }

    if( bala.position.x <= 0  ){
        resetVariables();
    }
    
    if( modoAuto ==false  && bala.position.x > 0 ){

        datosEntrenamiento.push({
                'input' :  [despBala , velocidadBala],
                'output':  [estatusAire , estatuSuelo ]  
        });

        console.log("Desplazamiento Bala, Velocidad Bala, Estatus, Estatus: ",
            despBala + " " +velocidadBala + " "+ estatusAire+" "+  estatuSuelo);
   }

}




function disparo(){
    velocidadBala =  -1 * velocidadRandom(300,800);
    bala.body.velocity.y = 0 ;
    bala.body.velocity.x = velocidadBala ;
    balaD=true;
}

function colisionH(){
    pausa();
}

function velocidadRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render(){

}

```
