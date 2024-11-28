document.getElementById("query-form").addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent page refresh
    const query = document.getElementById("search-query").value;

    // Clear previous results
    const status = document.getElementById("status");
    const suggestion = document.getElementById("suggestion");
    status.textContent = "Processing your query...";
    suggestion.classList.add("hidden");

    // Make API call to backend
    try {
        const response = await fetch("http://localhost:5000/evaluate-query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query }),
        });

        if (response.ok) {
            const data = await response.json();
            status.textContent = data.isWellFormed
                ? "Your search query is well-formed."
                : "Your search query is not well-formed.";

            // Display suggestion if not well-formed
            if (!data.isWellFormed) {
                suggestion.textContent = `Suggested query: ${data.suggestedQuery}`;
                suggestion.classList.remove("hidden");
            }
        } else {
            throw new Error("Failed to fetch response from server.");
        }
    } catch (error) {
        status.textContent = "An error occurred. Please try again.";
        console.error(error);
    }
});
