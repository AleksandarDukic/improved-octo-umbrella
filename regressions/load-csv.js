const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
  const headers = _.first(data);

  const indexes = _.map(columnNames, column => headers.indexOf(column));
  const extracted = _.map(data, row => _.pullAt(row, indexes));

  return extracted;
}

module.exports = function loadCSV(
  filename,
  {
    dataColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = false
  }
) {

  let data = fs.readFileSync(filename, { encoding: 'utf-8' }); // typeof data = "string"
  data = _.map(data.split('\n'), d => d.split(','));           // typeof data = "object"

  data = _.dropRightWhile(data, val => _.isEqual(val, ['']));  // brise sve objekte [''] koji su prazni stringovi i koji se nalaze na kraju niza ...[..][''][...][''][''] (samo poslednja dva)

  const headers = _.first(data);  // Converts the first character of string to lower case --- takes array of strings of labels ----['passedemissions','mpg','cylinders','displacement','horsepower','weight','acceleration','modelyear','carname']

  data = _.map(data, (row, index) => {
    if (index === 0) {
      return row;                   // leaves first row ----  ['passedemissions','mpg','cylinders','displacement','horsepower','weight','acceleration','modelyear','carname']
    }
    return _.map(row, (element, index) => { // this function either returns a parsedFloat result or a converted value
      if (converters[headers[index]]) {       // "converters" function must be the same name as the name of the category(column)
        const converted = converters[headers[index]](element); // does the converter function on the specified filed
        return _.isNaN(converted) ? element : converted;
      }
      const result = parseFloat(element.replace('"', '')); // destroys " character and tries to turn the result to number
      return _.isNaN(result) ? element : result;           // if the result is a number update the row if not (like car name) disgards result it and keeps the old value in the field of the object of specified index
    });
  });

  let labels = extractColumns(data, labelColumns);        // extract label
  data = extractColumns(data, dataColumns);               // turns to features
  
  data.shift();         // removes the first element ( the name of the column ) leaves only values
  labels.shift();

  if (shuffle) {
    data = shuffleSeed.shuffle(data, 'phrase');
    labels = shuffleSeed.shuffle(labels, 'phrase');
  }

  if (splitTest) {                        // if "splitTest" exist divides the data: (0,splitTest) - Test data, (splitTest, EOL) - Training data
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize)
    };
  } else {
    return { features: data, labels };
  }
};
