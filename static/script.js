document.getElementById('predictForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const data = {
        'PM2.5': document.getElementById('pm25').value,
        'PM10': document.getElementById('pm10').value,
        'NO': document.getElementById('no').value,
        'NO2': document.getElementById('no2').value,
        'NOx': document.getElementById('nox').value,
        'NH3': document.getElementById('nh3').value,
        'CO': document.getElementById('co').value,
        'SO2': document.getElementById('so2').value,
        'O3': document.getElementById('o3').value,
        'Benzene': document.getElementById('benzene').value,
        'Toluene': document.getElementById('toluene').value,
        'Xylene': document.getElementById('xylene').value,
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Map AQI category to image URLs
        const categoryImages = {
            good: 'https://img.freepik.com/premium-vector/cute-boy-breathing-fresh-air-vector-illustration_1334819-9291.jpg',
            moderate: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5Yixtzs12a5TKhO9_29HmDitk6A1qhbXWIg&s', // Provide the correct URL for moderate category
            poor: 'https://thumbs.dreamstime.com/b/boy-wearing-n-mask-dust-pm-air-pollution-young-men-breath-protection-safe-face-fog-danger-dirty-smog-sick-disease-protect-144323978.jpg',
            very_poor: 'https://c8.alamy.com/comp/2B1A5TF/city-air-pollution-smog-pollutants-suffocation-environment-and-passer-in-breathing-masks-vector-illustration-2B1A5TF.jpg',
            severe: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIyHLypGfoD6ORhunz9tiUk9YUDCRvPyDyXoM3j3f1NqKXH_oFVVH937hQ7stxbXR5nTM&usqp=CAU',
        };

        // Get the image URL based on the category, with a fallback
        const imageSrc = categoryImages[result.Category.toLowerCase()] || 'https://example.com/default_image.jpg';

        // Display the result along with the category image
        document.getElementById('output').innerHTML = `
            <div class="result">
                <h2>AQI: ${result.AQI}</h2>
                <p>Category: <span class="category ${result.Category.toLowerCase()}">${result.Category}</span></p>
                <img src="${imageSrc}" alt="${result.Category} image" class="category-image">
            </div>
        `;
    } catch (error) {
        console.error('Error:', error);
    }
});
