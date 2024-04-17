"use client"
import Head from 'next/head';
import styles from './chat.module.css';
import {Box,Button} from '@mui/material';
import { useState } from "react";
import RealSearchBar from "../../components/real_input_comp"
import axios from 'axios';
import { useEffect } from 'react';
import {HmacSHA256, enc} from 'crypto-js'

function CookiesComponent ({giveCookieConsent}) {
    return (

        <form className = {styles.cookiesbox}>
            <p>
                We use cookies and get your intial general location to save your message history and provide Mendy context to better understand you!
                Don't worry, you can opt in and its totally up to you. You can still have a great experience
                without it.
            </p>
            <Button size="medium" onClick = {() => giveCookieConsent(true)}>
            Accept
            </Button>
            <Button size="medium" onClick = {() => giveCookieConsent(false)}>
            Decline
            </Button>

        </form>


    )
}

const ChatBubbles = ({init_chat_hist}) => Object.entries(init_chat_hist).map(entry => (
        <div>
          <br></br>
          <div className={styles.humanChat}>
            {entry[0]}
            <br></br>
            {entry[1]['HumanMessage']}
          </div>
          <br></br>
          <div className={styles.aiChat}>
            {entry[0]}
            <br></br>
            {entry[1]['AIMessage']}
          </div>
          <br></br>
        </div>
      ));



export default function chat() {

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
      axios.get('http://127.0.0.1:8000/ai/setcookies/', JWT_options)
      .then(res => {
        //console.log(res['data']['response'])
        let resp = res['data']['response']
        if (value == "false"){
          window.localStorage.setItem("MENDY_SESSION", resp)
          console.log("new token after req: ",window.localStorage.getItem("MENDY_SESSION"))
        }else{
          if (resp != "FALSE"){
            //window.localStorage.setItem("MENDY_SESSION_CHAT_HIST", resp)
            console.log("yes!")
            console.log("response",resp)
            set_init_chat_hist(resp)
            Object.entries(resp).map(entry => (
              console.log(entry[0])
            ))
            

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

  //temporary, this variable will store the last valid server return value, the 
  //final version must store all responses 
  const [_temp, _set_temp] = useState("");
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
      axios.get('http://127.0.0.1:8000/ai/query', JWT_options)
      .then(res => {
        //console.log(res['data']['response'])
        let resp = res['data']['response'] //[currentTime, {"AIMessage" : resp , "HumanMessage" : query}]
        if (value == "false"){
          
          console.log(window.localStorage.getItem("MENDY_SESSION"))
        }else{
          //window.localStorage.setItem("MENDY_SESSION_CHAT_HIST", resp)
          console.log("YES!")
          init_chat_hist[resp[0]] = {"AIMessage" : resp[1]["AIMessage"] , "HumanMessage" : resp[1]["HumanMessage"] }
          set_init_chat_hist(init_chat_hist)
        }
        _set_temp(resp)
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
    if(typeof window !== 'undefined' && window.localStorage  && window.localStorage.getItem("MENDY_CONSENT") == "true"){ 
      setCookieComponent(null)
      sendJWT()
    
    }
    return () => {
      //unmount code
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  },[]);


  return (


    <div>
      <Head>
        <title>Main Page</title>
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
      <RealSearchBar handleClick={handleClick} searchQuery={searchQuery} setSearchQuery={setSearchQuery} btnDisabled={btnDisabled} setBtnDisabled={setBtnDisabled} />  

      </main>
    </div>

  );


}