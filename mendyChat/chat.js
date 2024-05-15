"use client"
import Head from 'next/head';
import styles from './chat.module.css';
import {Box,Button} from '@mui/material';
import { useState } from "react";
import RealSearchBar from "../components/real_input_comp"
import axios from 'axios';
import { useEffect } from 'react';
import {HmacSHA256, enc} from 'crypto-js'
/*Thank you for using our website, https://mndyrr.ai and interacting with our AI chatbot, Mendy™. We take your privacy seriously, and we want you to understand how we collect, use, and protect your personal information.
Our AI chatbot is designed to provide general information and support on various topics related to your overall well-being. However, please note that the information provided by the chatbot is not intended to be a substitute for professional medical advice, diagnosis, or treatment. The chatbot is not designed to provide medical, medication, or diagnostic advice, and you should not rely on the information provided by the chatbot as a substitute for professional advice.
Please be aware that the chatbot is an automated system, and it may not always provide 100% accurate information. While we strive to provide accurate information, we cannot guarantee the accuracy, completeness, or timeliness of the information provided by the chatbot, but it will produce better answers as a learning bot.
It is important to note that MNDYRR [mender] Technologies, Inc. is a lived experience organization, and we do not currently provide any professional services or help if people are in a crisis or have an urgent inquiry. If you require immediate assistance or are experiencing a crisis, please seek professional medical assistance or contact a mental health crisis hotline in your area.
We may collect personal information, such as your name, number, email address, and location when you interact with our chatbot. We will only use this information for the purposes of providing support and responding to your inquiries. We will not share your personal information with third parties unless we have your consent or are required by law to do so.
By using our website and interacting with our chatbot, you agree to the terms of this Privacy Notice and Disclaimer. If you have any questions or concerns about our privacy practices, please contact us at info@mndyrr.com
*/
/* <div class="buttonContainer"></div> */
function CookiesComponent ({giveCookieConsent}) {
    return (
      <div className = {styles.cookiesbox}>
        <form>
            <p>
            Thank you for using our website, <a href="https://mndyrr.ai">https://mndyrr.ai</a> and interacting with our AI chatbot, Mendy™. We take your privacy seriously, and we want you to understand how we collect, use, and protect your personal information.
            Our AI chatbot is designed to provide general information and support on various topics related to your overall well-being. However, please note that the information provided by the chatbot is not intended to be a substitute for professional medical advice, diagnosis, or treatment. The chatbot is not designed to provide medical, medication, or diagnostic advice, and you should not rely on the information provided by the chatbot as a substitute for professional advice.
            <br></br><br></br>
            Please be aware that the chatbot is an automated system, and it may not always provide 100% accurate information. While we strive to provide accurate information, we cannot guarantee the accuracy, completeness, or timeliness of the information provided by the chatbot, but it will produce better answers as a learning bot.
            It is important to note that MNDYRR [mender] Technologies, Inc. is a lived experience organization, and we do not currently provide any professional services or help if people are in a crisis or have an urgent inquiry. If you require immediate assistance or are experiencing a crisis, please seek professional medical assistance or contact a mental health crisis hotline in your area.
            <br></br><br></br>We may collect personal information, such as your name, number, email address, and location when you interact with our chatbot. We will only use this information for the purposes of providing support and responding to your inquiries. We will not share your personal information with third parties unless we have your consent or are required by law to do so.
By using our website and interacting with our chatbot, you agree to the terms of this Privacy Notice and Disclaimer. If you have any questions or concerns about our privacy practices, please contact us at info@mndyrr.com
            </p>
        </form>
        <div>
                      
          <Button size="medium" onClick = {() => giveCookieConsent(true)}>
              Accept
          </Button>
          <Button size="medium" onClick = {() => giveCookieConsent(false)}>
            Decline
          </Button>
        </div>
      </div>

    )
}

const ChatBubbles = ({init_chat_hist}) => Object.entries(init_chat_hist).map(entry => (
        <div key={entry[0]}>
          <span className = {styles.date}>{entry[0].split("-")[1]}/{entry[0].split("-")[2]}/{entry[0].split("-")[0]} {entry[0].split("-")[3].length == 1 ? "0" + entry[0].split("-")[3] : entry[0].split("-")[3]} : {entry[0].split("-")[4].length == 1 ? "0" + entry[0].split("-")[4] : entry[0].split("-")[4]}</span>
          <br></br>
          <div className={styles.humanChat}>
            <strong>You</strong>
            <br></br>
            {entry[1]['HumanMessage']}
          </div>

          <div className={styles.aiChat}>
            <br></br>
            <br></br>
            <strong>Mendy</strong>
            <br></br>
            {entry[1]['AIMessage']}
          </div>
          <br></br>
        </div>
      ));



export default function Chat() {

  function sendJWT(){
    if(typeof window !== 'undefined' && window.localStorage ){
      // getCookies();
      console.log("yes window")
      let value = window.localStorage.getItem("MENDY_SESSION") || "false"
      console.log("prev jwt: ", value)
      setCookie_data(value)
      //set the location
      let JWT_options = {
        params : {

          JWT : value,



        }
      }
      axios.get('http://0.0.0.0:8000/ai/setcookies/', JWT_options)
      .then(res => {
        //console.log(res['data']['response'])
        let resp = res['data']['response']
        if (value == "false"){
          window.localStorage.setItem("MENDY_SESSION", resp)
          console.log("new token after req: ",window.localStorage.getItem("MENDY_SESSION"))
        }else{
          if (resp != "FALSE"){
            //window.localStorage.setItem("MENDY_SESSION_CHAT_HIST", resp)
            console.log("cookie!")
            console.log("response",resp)
            set_init_chat_hist(resp)
            // Object.entries(resp).map(entry => (
            //   console.log(entry[0])
            // ))
            

          }else{
            console.log("AUTH FAILED")
          }
          
        }
      })    
      .catch(err => { 
        console.log(err) 
      })

      
    }else{
      console.log("no window")
    }
  }
  //DEMO SECTION
  //add a disclaimer popup before use of the chat bot section
  //https:/www.dualdiagnosis.org.uk/chatbot-disclaimer/
  const [searchQuery, setSearchQuery] = useState("");

  const [btnDisabled, setBtnDisabled] = useState(true);

  const [cookies_consent , setCookie] = useState(undefined)
  const [cookies_data, setCookie_data] = useState([])
  const [cookies_msg_data, set_msg_data] = useState([])
  const [geolocation, set_geolocation] = useState("")

  const [init_chat_hist, set_init_chat_hist] = useState(null)

  const [cookiepopup, setCookieComponent] = useState(<CookiesComponent giveCookieConsent={giveCookieConsent}/>)
  function handleClick(){
    //console.log("cookie settings: ", cookies.cookieConsent)
    //console.log(axios.defaults.withCredentials)
    if(typeof window !== 'undefined' && window.localStorage ){
      // getCookies();
      console.log("yes window")
      let value = window.localStorage.getItem("MENDY_SESSION") || "false"
      console.log("JWT TOKEN : ",value)
      console.log("geolocation: ",geolocation)
      let JWT_options = {
        params : {

          JWT : value,
          question : searchQuery,
          geolocation : geolocation
        }
      }
      axios.get('http://0.0.0.0:8000/ai/query', JWT_options)
      .then(res => {
        //console.log(res['data']['response'])
        let resp = res['data']['response'] //[currentTime, {"AIMessage" : resp , "HumanMessage" : query}]
        if (value == "false"){
          
          console.log(window.localStorage.getItem("MENDY_SESSION"))
        }else{
          //window.localStorage.setItem("MENDY_SESSION_CHAT_HIST", resp)
          console.log("on handle click:", resp)

          let newChats = {}
          newChats[resp[0]] = {"AIMessage" : resp[1]["AIMessage"] , "HumanMessage" : resp[1]["HumanMessage"] }
          if (init_chat_hist != null){
            Object.entries(init_chat_hist).map(entry => (
              newChats[entry[0]] = {'HumanMessage' : entry[1]['HumanMessage'], "AIMessage" :  entry[1]["AIMessage"]}

            ))
          }
          
          
          set_init_chat_hist(newChats)

        }
      })    
      .catch(err => { 
        console.log(err) 
      })

      
    }else{
      console.log("no window")
    }

  }


  function giveCookieConsent(consent) {
    //set location if consent == true
    if (consent == true){

      // navigator.geolocation.getCurrentPosition((position) => {
      //   console.log(position.coords.latitude ," ", position.coords.longitude)
      //   let location = JSON.stringify(position.coords.latitude) + "° N, " + JSON.stringify(position.coords.longitude) + "° W"
      //   //encrypt it HMACSHA256 then baseURL encode
      //   console.log(location)
      //   console.log("geosalt: ", process.env.GEO_SALT)
      //   let hmac = HmacSHA256(location, process.env.GEO_SALT);
      //   let encoded_location = hmac.toString(enc.Base64)
      //   //CryptoJS.HmacSHA256()
      //   set_geolocation(encoded_location);
      // });
     
    }
    setCookie(consent)
    window.localStorage.setItem("MENDY_CONSENT", JSON.stringify(consent))
    console.log("cookie settings: ", cookies_consent)
    setCookieComponent(null)
    let localstore_item = window.localStorage.getItem("MENDY_CONSENT")
    console.log("localstore: ", localstore_item)
    if(localstore_item == "true"){
      console.log("first time send")
      sendJWT()
    }


  }
  console.log("on start: ",cookies_consent)

  // const getCookies = () => {
  //   //console.log(localStorage.length)
  //   //console.log(localStorage.length)
  //   value = localStorage.getItem("MENDY_SESSION") || ""
  //   console.log("val: ",value)
  //   setCookie_data(value)
  // };

  useEffect(() => {
    // now access your localStorage
    const handleBeforeUnload = (event) => {
      if(typeof window !== 'undefined' && window.localStorage){
        event.preventDefault();
        window.localStorage.removeItem("MENDY_CONSENT") //for testing only maybe if session exists then it persists
        //window.localStorage.removeItem("MENDY_SESSION")
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    if(typeof window !== 'undefined' && window.localStorage && window.localStorage.getItem("MENDY_CONSENT") == "true"){ 
      setCookieComponent(null)
      //sendJWT()
    
    }
    return () => {
      //unmount code
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  },[init_chat_hist]);


  return (


    <div>
      <Head>
        <title>Chat</title>
        <link rel="icon" href="/favicon.ico" />

      </Head>

      <main>
        
        {
          cookiepopup
        }

        <Box color="black" className= {styles.mainbox}> 

        {
          init_chat_hist? <ChatBubbles init_chat_hist={init_chat_hist}/> : null
        }
      </Box> 
      <br></br>
      <RealSearchBar handleClick={handleClick} searchQuery={searchQuery} setSearchQuery={setSearchQuery} btnDisabled={btnDisabled} setBtnDisabled={setBtnDisabled} />  

      </main>
    </div>

  );


}