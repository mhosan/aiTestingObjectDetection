const axios = require('axios');
const fs = require('fs');
const path = require('path');

async function query(filename) {
	const data = fs.readFileSync(filename);
	const response = await fetch(
		"https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
		{
			headers: {
				Authorization: "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
				 "Content-Type": "application/octet-stream",
			},
			method: "POST",
			body: data,
		}
	);
	const result = await response.json();
	return result;
}

query("./input/piscina_satelite.jpg").then((response) => {
	console.log(JSON.stringify(response));
});