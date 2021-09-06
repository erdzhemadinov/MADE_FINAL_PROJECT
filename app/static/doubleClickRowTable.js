	const table = document.querySelector('table');
	const url = "/catch_query_interpretation/";
	
	// Отправка POST запроса:
	async function postData(url = '', data = {}) {
		const response = await fetch(url, {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify(data)
		});
		if (!response.ok) {
			// Сервер вернул код ответа за границами диапазона [200, 299]
			return Promise.reject(new Error(
				'Response failed: ' + response.status + ' (' + response.statusText + ')'
			));	
		}
		location.replace("/output_frame_interpretation/");
	}
	
	table.addEventListener('dblclick', function (e) {
		let cell = e.target;
		if (cell.tagName.toLowerCase() != 'td')
			return;
		/*let i = cell.parentNode.rowIndex;
		let j = cell.cellIndex;*/
		let tableRow = cell.parentNode.cells;

		var dataQA = {
			question: tableRow.item(1).innerHTML,
			answer: tableRow.item(2).innerHTML,
			prob: tableRow.item(3).innerHTML
		};
		postData(url, dataQA)
	});