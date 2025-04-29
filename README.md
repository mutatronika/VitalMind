# 🧠 VitalMind  
**Agente de IA para monitoreo y atención médica personalizada**

VitalMind es un agente inteligente diseñado para apoyar el seguimiento de salud de pacientes con necesidades especiales debido a su edad o condición médica. Utiliza modelos de lenguaje avanzados (Azure OpenAI), capacidades de razonamiento con datos de salud y almacenamiento semántico para brindar respuestas útiles y recomendaciones.

---

## 🚀 Características principales

- Análisis de datos de salud (presión arterial, glucosa, estrés, etc.)
- Consulta conversacional natural usando LLMs
- Integración con Semantic Kernel y Azure OpenAI
- Soporte para plugins personalizados (como BloodPressurePlugin)

--------------

Jerarquia de archivos

Estructura del repositorio


.vscode/
.github/
data/                     # Archivos de datos como BloodPressuredataset.csv
docs/                     # Documentación adicional
notebooks/                # Notebooks Jupyter como agentia.ipynb, autoagent.ipynb
src/                      # Código fuente principal
    plugins/              # Plugins personalizados como BloodPressurePlugin
    pipelines/            # Scripts relacionados con pipelines como BP_pipelines.py
    main.py               # Archivo principal para ejecutar el proyecto
tests/                    # Pruebas unitarias y de integración
.env
.env.example
requirements.txt
README.md
LICENSE

-----------------------

## ⚙️ Configuración del proyecto

### 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/VitalMind.git
cd VitalMind
```

### 2. Copiar variables de entorno
https://github.com/settings/personal-access-tokens
```
cp .env.example .env
```
### 3. Crear y activar entorno virtual
```
python3 -m venv venv
source venv/bin/activate
```

En Windows (PowerShell)
Primero, habilita la ejecución de scripts si ves errores:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

python -m venv venv
.\venv\Scripts\activate
```


### 4. Instalar dependencias    
```
python.exe -m pip install --upgrade pip

pip install -r requirements.txt

```

