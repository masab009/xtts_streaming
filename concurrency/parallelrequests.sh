#!/bin/bash

texts=(
    "In Verdant Valley, Elara charted forgotten trails."
    "Under the moon’s glow, she began her quest."
    "The map was a beacon in the night."
)

send_request() {
    i=$1
    text=$2
    echo "Sending request $i: $text"
    if curl -X POST "http://localhost:8000/tts/stream" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"language\": \"en\", \"chunk_size\": 20, \"output_file\": \"server_output_$i.wav\"}" \
        --max-time 300 -o "local_output_$i.wav" 2>/dev/null; then
        echo "Completed request $i, saved as local_output_$i.wav (server: server_output_$i.wav)"
    else
        echo "Failed request $i" >&2
        return 1
    fi
}

export -f send_request

# Run 2 requests in parallel to avoid GPU overload
for i in "${!texts[@]}"; do
    echo "$((i+1)) ${texts[$i]}"
done | parallel -j 2 --colsep ' ' send_request

echo "All requests completed. Check local directory for local_output_*.wav and server for server_output_*.wav"