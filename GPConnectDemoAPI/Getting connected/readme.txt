Server: https://systmoneukdemo1.tpp-uk.com
Port: 443

The URL is different for each type of message, but they all hit the server above. 
E.g.
  https://systmoneukdemo1.tpp-uk.com/SystmOneMHS/NHSConnect/Z12345/STU3/1/Appointment
  https://systmoneukdemo1.tpp-uk.com/SystmOneMHS/NHSConnect/Z12345/STU3/1/Patient/$gpc.getstructuredrecord

The API uses TLS with mutual authentication, so as well as trusting the GpConnect CA, you need to use the certificate and private key in HackCambridgeDemoUser.pfx
pw: cambridge

Messages require several HTTP headers to work. These are defined in the specs, but in general you can use the headers from the example messages we've provided.
Most of them (ssp-...) are to do with routing messages around inside the NHS network, Just use the fixed values provided for the Demo API.

Message requests and responses must be in JSON.
Responses use gzip and HTTP chunking. Most HTTP libraries should just support this out of the box.
