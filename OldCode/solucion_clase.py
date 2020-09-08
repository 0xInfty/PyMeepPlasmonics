import numpy as np
import matplotlib.pyplot as plt
import imageio
###############################################################################   
def calcular_aceleracion(delta_x,delta_y):
    G = 6.693E-11
    m_sol = 1.989e30
    suma_deltas_cuadrado = (delta_x**2 + delta_y**2)
    aceleracion_x=(G * m_sol * delta_x) / (suma_deltas_cuadrado * np.sqrt(suma_deltas_cuadrado))
    aceleracion_y=(G * m_sol * delta_y) / (suma_deltas_cuadrado * np.sqrt(suma_deltas_cuadrado))
    return aceleracion_x,aceleracion_y

def hacer_foto(X,Y,aceleracion_x,aceleracion_y,dia):#grafico 1 foto, i controla el color
    plt.clf() #limpio la figura anterior
    plt.plot(X,Y,'grey') #grafico toda la trayectoria con una linea gris
    plt.plot(X[dia],Y[dia],'bo',ms=10) #grafico la tierra
    plt.arrow(X[dia], Y[dia], aceleracion_x[dia]*10**12.5, aceleracion_y[dia]*10**12.5,width=10**9.5,Color='g')  #grafico la fuerza
    plt.plot(0,0,'yo',ms=30) #grafico el sol
    plt.title('Dia '+str(i)) #pongo el nro de dia como titulo
    plt.show()
    
def hacer_video(X,Y,f_x,f_y,nombre_gif):
    plt.figure() #abro una nueva figura para ir haciendo las fotos
    fotos=[] #aca voy a ir guardando las fotos
    for i in range (len(X)):
        if i%2==0: # para guardar 1 de cada 2 fotos
            hacer_foto(X,Y,f_x,f_y,i) #grafico una foto
            rango_X=np.max(X)-np.min(X);rango_Y=np.max(Y)-np.min(Y) # defino el rango para fijar los ejes x e y todo el video, que incluyan toda la trayectoria
            plt.axis([np.min(X)-rango_X*0.1, np.max(X)+rango_X*0.1, np.min(Y)-rango_Y*0.1, np.max(Y)+rango_Y*0.1]) #fijo los ejes en todo el video
            plt.savefig(nombre_gif+'_foto.png') #la foto se guarda para hacer el gif
            fotos.append(imageio.imread(nombre_gif+'_foto.png')) #lee la foto guardada y la agrega a la lista de fotos
            print(str(i)+' de '+str(len(X))) #impirmo un contador porque tarda un poco
    imageio.mimsave(nombre_gif+'_video.gif', fotos) #crea el fig a partir de la lista de fotos, con el nombre indicado en el directorio
    print('Video Guardado')
###############################################################################   
m_sol = 1.989e30
m_tierra = 5.972e24
G = 6.693E-11 # constante de la fuerza gravitatoria en m^3/s^/kg
x_sol = 0
y_sol = 0

#mov real (https://solarsystem.nasa.gov/solar-system/sun/by-the-numbers/)
x_tierra = [-147095000000.0, -147095000000.0] #posiciones x e y de la tierra en metros
y_tierra = [2617920000.0, 0.0]

##hiperbola
#x_tierra = [-147095000000.0, -147095000000.0]
#y_tierra = [4000000000.0, 0.0]

##elipse
#x_tierra = [-147095000000.0, -147095000000.0]
#y_tierra = [2000000000.0, 0.0]

dias = [-1, 0]
dt = 60 * 60 * 24 #paso temporal en segundos
tiempo_total = 400  #número de pasos (dias) de la simulación


aceleracion_x=[0]
aceleracion_y=[0]
for i in range(1, tiempo_total - 1):
    #calculo los deltas
    delta_x = x_sol - x_tierra[i]
    delta_y = y_sol - y_tierra[i]
    
    a_x,a_y=calcular_aceleracion(delta_x,delta_y)
    
    aceleracion_x.append(a_x)
    aceleracion_y.append(a_y)

    #actualizo la posicion x
    x_actual = x_tierra[i]
    x_prev = x_tierra[i-1]

    x_nueva = 2 * x_actual - x_prev + aceleracion_x[i] * dt**2
    x_tierra.append(x_nueva)

    #actualizo la posicion y
    y_actual = y_tierra[i]
    y_prev = y_tierra[i-1]

    y_nueva = 2 * y_actual - y_prev + aceleracion_y[i] * dt**2
    y_tierra.append(y_nueva)

    #actualizo el tiempo
    dias.append(i)

plt.figure(0)
hacer_foto(x_tierra,y_tierra,aceleracion_x,aceleracion_y,i)

nombre_gif='elipse' #nombre con el que se guardara en la carpeta donde esta el programa
hacer_video(x_tierra,y_tierra,aceleracion_x,aceleracion_y,nombre_gif)
