# 🧠 VitalMind  
**Agente de IA para monitoreo y atención médica personalizada**

VitalMind es un agente inteligente diseñado para apoyar el seguimiento de salud de pacientes con necesidades especiales debido a su edad o condición médica. Utiliza modelos de lenguaje avanzados (Azure OpenAI), capacidades de razonamiento con datos de salud y almacenamiento semántico para brindar respuestas útiles y recomendaciones.

---

## 🚀 Características principales

- Análisis de datos de salud (presión arterial, glucosa, estrés, etc.)
- Consulta conversacional natural usando LLMs
- Integración con Semantic Kernel y Azure OpenAI
- Soporte para plugins personalizados (como BloodPressurePlugin)

---



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
```
python -m venv venv
<<<<<<< HEAD
.\venv\Scripts\activate
=======
venv\Scripts\activate
>>>>>>> upstream/main

python3 -m venv venv
source venv/bin/activate

```
### 4. Instalar dependencias    
```
pip install -r requirements.txt
```
