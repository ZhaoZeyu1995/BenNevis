

<!-- ![Icon](icon1.jpeg) -->
<div align="center">
<img src="icon.jpeg" width="200" height="200" alt="Alt text" title="Optional title">
</div>


# Introducción

Esta es una herramienta (toolkit) de Reconocimiento Automático de Voz (ASR) de extremo a extremo (E2E) basada en un Transductor de Autómata Finito Ponderado Diferenciable (DWFST) con soporte para la definición flexible de topología, llamada **BenNevis**, que está implementada principalmente con [PyTorch](https://github.com/pytorch/pytorch) y [k2](https://github.com/k2-fsa/k2), donde el primero actúa como el backend de red neuronal y el segundo para DWFST.

**Nota: este proyecto aún está en desarrollo, por lo que el código podría cambiar rápidamente en el futuro**

# Características

* Soporte flexible de **topología**
* Manipulación de datos al estilo Kaldi con [kaldiio](https://github.com/nttcslab-sp/kaldiio) y scripts de Kaldi
* DWFST respaldado por [k2](https://github.com/k2-fsa/k2)
* Implementación puramente en [PyTorch](https://github.com/pytorch/pytorch)
* Soporte para Distributed Data Parallel (DDP) con `torch.distributed`
* Soporte para el registrador (logger) [WandB](https://wandb.ai/)

# Instalación

## Kaldi

Primero debes instalar [Kaldi](https://github.com/kaldi-asr/kaldi) y luego ejecutar 
```shell
cd BenNevis/tools
./put_kaldi.sh /path/to/kaldi
```
Este script simplemente crea un enlace simbólico a tu Kaldi.

En BenNevis, Kaldi se utiliza para la preparación de datos, la extracción de características, la compilación de grafos (basado en OpenFST) y la decodificación basada en WFST (decodificador `decode-faster`).

## Entorno Virtual

Debes crear un entorno virtual, llamado `venv`, dentro de `BenNevis/tools/` para ejecutar el código de BenNevis.
Existen dos formas de crear el entorno virtual para ejecutar BenNevis.
### Script Automático

Puedes simplemente ejecutar
```shell
cd BenNevis/tools
./create_env.sh
```
para crear el entorno correspondiente.
Puedes notar que las versiones de `PyTorch` y `k2` están codificadas en el script.
Sin embargo, siéntete libre de cambiarlas a tus versiones preferidas.

### Configuración Manual

Se recomienda principalmente el script automático, pero también puedes hacerlo de manera manual.
```shell
cd BenNevis/tools/
python3 -m venv venv
```
Actualmente, se recomienda ``Python >= 3.8``.
De hecho, puedes elegir cualquier versión de Python siempre que puedas instalar `PyTorch` y `k2` correctamente.

A partir de ahora, necesitarás activar este entorno virtual e instalar librerías mediante `pip`.
Asumiendo que ahora estás en `BenNevis/tools/`,
```shell
source venv/bin/activate
```

#### PyTorch
A continuación se muestra la versión que estoy utilizando.
Siéntete libre de usar otras versiones, pero al menos debe soportar [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html).
Además, asegúrate de que tu PyTorch sea compatible con CUDA.
```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### k2

La instalación de `k2` es un poco complicada, ya que debes asegurarte de que el `wheel` coincida con tu versión de `PyTorch`, incluyendo la versión de `CUDA`.
Por ejemplo, 
```Shell
pip install k2==1.24.4.dev20231220+cuda11.8.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html
```
Puedes ver que la versión de CUDA y torch coincide exactamente con lo que acabo de instalar.
Puedes consultar [esta página](https://k2-fsa.github.io/k2/cuda.html) para ver más `wheels` precompilados disponibles.

#### Misceláneos

A continuación se presentan otras librerías que necesitas instalar con pip. Ten en cuenta que algunas son opcionales.
No especifico las versiones para ellas, pero por lo general las últimas versiones deberían funcionar.
```shell
# for the progress bar support
pip install tqdm  
# for configuration management
pip install hydra-core 
# a very powerful logger
pip install wandb 
# for model summary display
pip install torchinfo 
# for kaldi-format data IO
pip install kaldiio 
# for whisper finetuning
pip install openai-whisper

# Currently Optional
pip install transformers
pip install datasets
pip install numba
pip install librosa
pip install soundfile
```

# Inicio Rápido

La mejor manera de comenzar es consultar la receta `egs/yesno/asr/run.sh`, donde podrás ver el flujo de trabajo básico en BenNevis.

Suele incluir

1. Preparación de datos
2. Extracción de características
3. Preparación del grafo de entrenamiento (para diversas topologías)
4. Preparación del grafo de decodificación (para diversas topologías)
5. Entrenamiento (basado en Distributed Data Parallel de PyTorch, consulta `run/train.py` para más detalles)
6. Predicción (obtención de las probabilidades posteriores del modelo NN, consulta `run/predict.sh` para más detalles)
7. Decodificación (utilizando el grafo de decodificación y las probabilidades posteriores como entradas, consulta `run/decode_faster.sh` para más detalles)
8. Alineación (opcional, admite alineación con ground truth o resultados de decodificación, consulta `run/align.sh` para más detalles)

# Contacto

No dudes en contactarme por [correo electrónico](mailto:zeyuhongwu1995@gmail.com) por cualquier consulta sobre BenNevis.
