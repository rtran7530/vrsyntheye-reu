# Generating .json files for quality evaluation using ISO standards
# USER CONFIGURATION - SET YOUR DIRECTORIES HERE
# 1. INPUT: Directory containing the images you want to evaluate (.png)
export INPUT_DIR="filtered_genuine_Unreflect"

# 2. OUTPUT: Directory where the resulting .json quality files will be saved
export OUTPUT_DIR="filtered_genuine_Unreflect_json"

# 3. LOG FILE: Text file to record any images that cause the tool to crash
export CRASH_LOG="poisoned_images_log_Unreflect.txt"

# 4. PROCESSING: Number of parallel CPU threads to use (adjust based on hardware)
export THREADS=12   

# Generating .json files for quality evaluation using ISO standards
mkdir -p "$OUTPUT_DIR"
find "$INPUT_DIR" -type f -name "*.png" | xargs -P "$THREADS" -I {} bash -c '
    img_path="{}"
    relative_path="${img_path#$INPUT_DIR/}"
    json_path="$OUTPUT_DIR/${relative_path%.png}.json"
    
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
        echo "[CRASH] Exit $EXIT_CODE on file: $img_path" >> "$CRASH_LOG"
        echo "🚨 SKIPPED POISONED FILE: $img_path"
    else
        echo "✅ Evaluated: $img_path"
    fi
'
