# Initialize the array
a=()

# Populate the array with numbers from 1 to 100
for ((i=1; i<=100; i++)); do
    a+=($i)
done

# Optionally, to verify the contents of the array
echo ${a[@]}
