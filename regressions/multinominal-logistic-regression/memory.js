// to examine memory usage in node we can run our node script using e.g. node --inspect-brk memory.js
// and then going to chrome at chrome://inspect, then clicking on your script- a debugger window
// will open and you can go up to where you placed your debugger by clicking on the blue arrow,
// then go to the memory tab, have heap selected, and then generate a memory usage report. this
// gives us a snapshot. the shallow size vs retained size diff: take an array for example. when
// you create an array two things are created: an object with all the familiar methods of the array that
// also has a referecne to the collection of values, and then there is the collection of values
// themselves. shallow size is what the object instance is using (the object with methods) and
// retained size is the collection.

// note that js garbage collection works e.g. on each line it will ask itself if it needs to store
// in memory and when it can clear, it does (e.g. a variable declared inside another scope and you
// are now outside the scope). so basically as long as the shallow object has a reference to a
// value, it will not be garbage collected.

// we will use this file along with the debugger and the chrome inspector to analyze how to
// optimize and manage memory usage. see differences in the memory report between commenting
// out the return statement below and not commenting it out.
const _ = require('lodash');

const loadData = () => {
  // makes an array with elements from first arg up to but not including second arg
  const randoms = _.range(0, 999999);

  return randoms; // comment out this line
};

const data = loadData();

debugger;
