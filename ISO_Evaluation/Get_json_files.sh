# Generating .json files for quality evaluation using ISO standards
mkdir -p filtered_genuine_Unreflect_json
find filtered_genuine_Unreflect -type f -name "*.png" | xargs -P 12 -I {} bash -c '
    img_path="{}"
    relative_path="${img_path#filtered_genuine_Unreflect/}"
    json_path="filtered_genuine_Unreflect_json/${relative_path%.png}.json"
    
    # IDEMPOTENCY: Skip files already completed
    if [ -f "$json_path" ]; then
        exit 0
    fi
    
    # Create the target directory
    json_dir=$(dirname "$json_path")
    mkdir -p "$json_dir"
    
    # Run BIQT, but swallow the output and catch the exit code
    biqt -p BIQTIris -f json -o "$json_path" "$img_path" > /dev/null 2>&1
    
    # Capture the exit code of the BIQT command
    EXIT_CODE=$?
    
    # If the exit code is not 0 (meaning it crashed or segfaulted)
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[CRASH] Exit $EXIT_CODE on file: $img_path" >> poisoned_images_log_Unreflect.txt
        echo "🚨 SKIPPED POISONED FILE: $img_path"
    else
        echo "✅ Evaluated: $img_path"
    fi
'
