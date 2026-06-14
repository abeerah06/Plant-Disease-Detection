
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "name": "Bacterial Spot",
        "healthy": False,
        "symptoms": "Small, water-soaked spots on leaves that enlarge into "
                    "dark brown, scabby lesions; severe cases cause leaf drop.",
        "treatment": "Remove and destroy infected debris. Apply copper-based "
                     "bactericides. Use disease-free seed and avoid overhead "
                     "irrigation to limit splash spread.",
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "name": "Healthy",
        "healthy": True,
        "symptoms": "No disease symptoms detected. Foliage is uniformly green "
                    "with no lesions or discoloration.",
        "treatment": "No action required. Maintain balanced watering, good air "
                     "circulation and routine monitoring.",
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "name": "Early Blight",
        "healthy": False,
        "symptoms": "Concentric 'target-board' brown rings on older leaves, "
                    "often surrounded by a yellow halo.",
        "treatment": "Apply chlorothalonil or mancozeb fungicides. Rotate crops, "
                     "remove volunteer plants and ensure adequate potassium "
                     "nutrition.",
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "name": "Late Blight",
        "healthy": False,
        "symptoms": "Large, dark, greasy-looking blotches on leaves with white "
                    "fungal growth on the underside in humid conditions.",
        "treatment": "Act fast — this disease destroyed crops in the Irish "
                     "Famine. Apply systemic fungicides (e.g. metalaxyl), "
                     "destroy infected plants and avoid wet foliage.",
    },
    "Potato___healthy": {
        "plant": "Potato",
        "name": "Healthy",
        "healthy": True,
        "symptoms": "No disease symptoms detected. Leaves are firm and green.",
        "treatment": "No action required. Continue good field hygiene and "
                     "monitor after rain or high humidity.",
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "name": "Bacterial Spot",
        "healthy": False,
        "symptoms": "Numerous small, dark, angular spots on leaves and fruit; "
                    "leaves may yellow and drop.",
        "treatment": "Use copper sprays, certified disease-free seed and crop "
                     "rotation. Avoid working with plants when foliage is wet.",
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "name": "Early Blight",
        "healthy": False,
        "symptoms": "Brown concentric-ring lesions on lower, older leaves; "
                    "progressive defoliation from the bottom up.",
        "treatment": "Apply mancozeb or chlorothalonil, mulch to prevent soil "
                     "splash, stake plants and remove infected lower leaves.",
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "name": "Late Blight",
        "healthy": False,
        "symptoms": "Irregular grey-green water-soaked patches that turn brown; "
                    "rapid collapse of foliage in cool, wet weather.",
        "treatment": "Remove and destroy infected plants immediately. Apply "
                     "protectant fungicides and improve airflow and drainage.",
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "name": "Leaf Mold",
        "healthy": False,
        "symptoms": "Pale yellow spots on the upper leaf surface with olive-"
                    "green to brown velvety mold underneath.",
        "treatment": "Reduce humidity and improve ventilation (especially in "
                     "greenhouses). Apply fungicides and space plants well.",
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "name": "Septoria Leaf Spot",
        "healthy": False,
        "symptoms": "Many small circular spots with dark borders and grey "
                    "centres, typically starting on lower leaves.",
        "treatment": "Remove infected leaves, mulch the soil, rotate crops and "
                     "apply fungicides at first sign of symptoms.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "name": "Two-Spotted Spider Mite",
        "healthy": False,
        "symptoms": "Fine yellow stippling and webbing on leaves; foliage turns "
                    "bronze and dries out under heavy infestation.",
        "treatment": "Spray with water to dislodge mites, apply insecticidal "
                     "soap or miticides, and introduce predatory mites.",
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "name": "Target Spot",
        "healthy": False,
        "symptoms": "Brown lesions with concentric rings on leaves and fruit, "
                    "resembling a target.",
        "treatment": "Apply fungicides, prune for airflow, and remove infected "
                     "plant material from the field.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomato",
        "name": "Yellow Leaf Curl Virus",
        "healthy": False,
        "symptoms": "Upward curling, yellowing and stunting of leaves; severely "
                    "reduced fruit set. Spread by whiteflies.",
        "treatment": "Control whitefly vectors with insecticides and reflective "
                     "mulch, use resistant cultivars and remove infected plants.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "name": "Mosaic Virus",
        "healthy": False,
        "symptoms": "Mottled light/dark green mosaic pattern, leaf distortion "
                    "and stunted growth.",
        "treatment": "No cure once infected — remove plants. Disinfect tools and "
                     "hands, control aphids and use resistant varieties.",
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "name": "Healthy",
        "healthy": True,
        "symptoms": "No disease symptoms detected. Leaves are green and intact.",
        "treatment": "No action required. Maintain consistent watering and "
                     "monitor regularly for early signs of disease.",
    },
    
    "split": {
        "plant": "—",
        "name": "Unknown / Non-leaf (data artifact)",
        "healthy": None,
        "symptoms": "This output corresponds to a spurious 'split' folder that "
                    "was accidentally included during training. It is not a real "
                    "plant disease class.",
        "treatment": "If this is the top prediction, the uploaded image is "
                     "likely not a clear single leaf. Try a closer, well-lit "
                     "photo of one leaf.",
    },
}


def get_info(raw_class: str) -> dict:
    """Return metadata for a raw class name, with a safe fallback."""
    return DISEASE_INFO.get(raw_class, {
        "plant": "Unknown",
        "name": raw_class.replace("_", " ").strip(),
        "healthy": None,
        "symptoms": "No description available.",
        "treatment": "No recommendation available.",
    })


def pretty_label(raw_class: str) -> str:
    """'Tomato_Early_blight' -> 'Tomato — Early Blight'."""
    info = get_info(raw_class)
    if info["plant"] in ("—", "Unknown"):
        return info["name"]
    return f"{info['plant']} — {info['name']}"
