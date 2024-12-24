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

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('API Result:', result);

    const category = result.Category || 'Unknown';
    const categoryKey = category.toLowerCase();
    console.log('Mapped Category Key:', categoryKey);

    

    document.getElementById('output').innerHTML = `
      <div class="result">
        <h2>AQI: ${result.AQI}</h2>
        <p>Category: <span class="category ${categoryKey}">${category}</span></p>
      
      </div>
    `;
  } catch (error) {
    console.error('Error:', error);
    document.getElementById('output').innerHTML = `
      <p style="color: red;">An error occurred: ${error.message}. Please try again.</p>
    `;
  }
});
