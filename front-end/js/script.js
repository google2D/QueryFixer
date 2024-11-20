async function fixQuery() {
    const query = document.getElementById("searchQuery").value;

    // Send the query to the backend API
    const response = await fetch("http://127.0.0.1:5000/fix-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
    });

    const data = await response.json();

    // Display the fixed query or an error
    if (data.fixed_query) {
        document.getElementById("fixedQuery").innerText = `Did you mean: ${data.fixed_query}?`;
    } else {
        document.getElementById("fixedQuery").innerText = `Error: ${data.error}`;
    }
}
